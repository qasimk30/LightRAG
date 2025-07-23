import asyncio
import logging
import os
import sys
from typing import Dict, Optional, Any
from urllib.parse import urlencode, parse_qs
from datetime import datetime, timedelta
import uuid
from pathlib import Path

from fastmcp import FastMCP
from extensions.app.config import initialize_config
from extensions.app.rag_manager import RAGManager
from extensions.app.mcp_tools import MCPTools

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, RedirectResponse, PlainTextResponse
from starlette.requests import Request
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import httpx
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_file = os.getenv("LOG_FILE")

logging_config = {
    'level': getattr(logging, log_level, logging.INFO),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

if log_file:
    logging_config['filename'] = log_file

logging.basicConfig(**logging_config)
logger = logging.getLogger(__name__)

# Store registered clients in memory (in production, use a database)
class RegisteredClient(BaseModel):
    client_id: str
    client_name: str
    redirect_uris: list[str]
    grant_types: list[str]
    response_types: list[str]
    scope: Optional[str] = None
    token_endpoint_auth_method: str = "none"
    created_at: float

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    scope: str
    expires_in: int
    refresh_token: Optional[str] = None

class KeycloakConfig:
    def __init__(self, realm_url: str, client_id: str, client_secret: str):
        self.realm_url = realm_url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_endpoint = f"{realm_url}/protocol/openid-connect/auth"
        self.token_endpoint = f"{realm_url}/protocol/openid-connect/token"
        self.userinfo_endpoint = f"{realm_url}/protocol/openid-connect/userinfo"
        self.introspect_endpoint = f"{realm_url}/protocol/openid-connect/token/introspect"

class OAuth2Server:
    def __init__(self, keycloak_config: KeycloakConfig, server_url: str):
        self.keycloak_config = keycloak_config
        self.server_url = server_url.rstrip('/')
        self.registered_clients: Dict[str, RegisteredClient] = {}
        self.http_client = httpx.AsyncClient()
    
    async def close(self):
        await self.http_client.aclose()
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token with Keycloak and return user info"""
        try:
            logger.debug(f"Verifying token: {token[:20]}...")
            
            # First try to get userinfo
            response = await self.http_client.get(
                self.keycloak_config.userinfo_endpoint,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            logger.debug(f"Userinfo response status: {response.status_code}")
            
            if response.status_code == 200:
                userinfo = response.json()
                logger.debug(f"Userinfo response: {userinfo}")
                return userinfo
            else:
                logger.debug(f"Userinfo failed with status {response.status_code}: {response.text}")
            
            # If userinfo fails, try token introspection
            logger.debug("Trying token introspection...")
            response = await self.http_client.post(
                self.keycloak_config.introspect_endpoint,
                data={
                    "token": token,
                    "client_id": self.keycloak_config.client_id,
                    "client_secret": self.keycloak_config.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            logger.debug(f"Introspection response status: {response.status_code}")
            
            if response.status_code == 200:
                introspect_result = response.json()
                logger.debug(f"Introspection result: {introspect_result}")
                if introspect_result.get("active"):
                    return introspect_result
                else:
                    logger.debug("Token is not active according to introspection")
            else:
                logger.debug(f"Introspection failed with status {response.status_code}: {response.text}")
            
            return None
            
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            import traceback
            traceback.print_exc()
            return None

class AuthAndWWWMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, oauth_server: OAuth2Server, server_url: str):
        super().__init__(app)
        self.oauth_server = oauth_server
        self.server_url = server_url.rstrip('/')

    async def dispatch(self, request: Request, call_next):
        # Paths we skip auth for
        skip_paths = [
            "/.well-known",
            "/register",
            "/authorize",
            "/token",
            "/userinfo",
            "/health",
            "/favicon.ico",
            "/debug",
        ]

        # Log incoming path
        logger.debug(f"[AuthAndWWW] >>> Incoming path: {request.url.path}")

        if any(request.url.path.startswith(path) for path in skip_paths):
            response = await call_next(request)
            logger.debug(f"[AuthAndWWW] <<< Skipped auth, status {response.status_code} for {request.url.path}")
            return response

        # Require auth for all other endpoints (including MCP tools)
        auth_header = request.headers.get("Authorization")
        logger.debug(f"[AuthAndWWW] Auth header for {request.url.path}: {auth_header}")

        if not auth_header or not auth_header.startswith("Bearer "):
            logger.error(f"[AuthAndWWW] Missing or invalid auth header for path: {request.url.path}")
            return self._unauthorized_response(request.url.path)

        token = auth_header[7:]  # strip Bearer
        logger.debug(f"[AuthAndWWW] Extracted token: {token[:20]}...")

        user_info = await self.oauth_server.verify_token(token)
        if not user_info:
            logger.error(f"[AuthAndWWW] Invalid token for path: {request.url.path}")
            return self._unauthorized_response(request.url.path)

        # Add user info to request state
        # request.state.user_info = user_info
        # request.state.access_token = token
        logger.info(f"[AuthAndWWW] Authenticated user: {user_info.get('preferred_username', 'unknown')} for path: {request.url.path}")

        # Let the request continue
        response = await call_next(request)
        logger.debug(f"[AuthAndWWW] <<< Normal response status {response.status_code} for {request.url.path}")
        return response

    def _unauthorized_response(self, path: str):
        # Build 401 response with proper WWW-Authenticate
        logger.debug(f"[AuthAndWWW] >>> Preparing 401 response with WWW-Authenticate for path: {path}")
        
        header_value = (
            'Bearer realm="MCP", '
            f'authorization="{self.server_url}/authorize", '
            f'token="{self.server_url}/token", '
            f'resource_metadata="{self.server_url}/.well-known/oauth-authorization-server"'
        )
        logger.debug(f"[AuthAndWWW] >>> Adding header: {header_value}")
        return JSONResponse(
            {"error": "Missing or invalid authorization header"},
            status_code=401,
            headers={"WWW-Authenticate": header_value},
        )

# Global variables to store server components
oauth_server: Optional[OAuth2Server] = None
keycloak_config: Optional[KeycloakConfig] = None

def get_config_path() -> str:
    """Get configuration file path from environment or default"""
    config_path = os.getenv("CONFIG_PATH")
    
    if config_path:
        return config_path
    
    # Try common locations
    possible_paths = [
        "/app/config.json",
        "/app/extensions/app/config.json",
        "./config.json",
        "./extensions/app/config.json"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If none found, use the original default but warn
    default_path = "/app/config.json"
    logger.warning(f"No config file found in standard locations, using: {default_path}")
    return default_path

async def run_server():
    """Run the server - this function is called within an async context"""
    global oauth_server, keycloak_config
    
    # Get configuration from environment variables
    config_path = get_config_path()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8091"))
    
    # Parse command line arguments (still supported for override)
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
        elif arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
            i += 2
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
            i += 2
        else:
            if not arg.startswith("-"):
                config_path = arg
            i += 1
    
    # Keycloak configuration from environment
    keycloak_realm_url = os.getenv("KEYCLOAK_REALM_URL")
    keycloak_client_id = os.getenv("KEYCLOAK_CLIENT_ID")
    keycloak_client_secret = os.getenv("KEYCLOAK_CLIENT_SECRET")
    server_url = os.getenv("SERVER_URL")
    enable_oauth_middleware = os.getenv("ENABLE_OAUTH_MIDDLEWARE")
    
    try:
        # Validate config file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Initialize configuration
        config = initialize_config(config_path)
        logger.info(f"Configuration loaded from: {config_path}")
        
        # Initialize Keycloak config
        keycloak_config = KeycloakConfig(
            realm_url=keycloak_realm_url,
            client_id=keycloak_client_id,
            client_secret=keycloak_client_secret
        )
        
        # Initialize OAuth2 server
        oauth_server = OAuth2Server(keycloak_config, server_url)
        
        # Initialize FastMCP
        mcp = FastMCP("Document-Ingestion-Server")
        
        # Initialize RAG Manager
        rag_manager = RAGManager(config)
        await rag_manager.initialize()
        
        # Initialize MCP Tools (this will register all tools with FastMCP)
        mcp_tools = MCPTools(mcp, config, rag_manager)

        # Add OAuth2 endpoints as custom routes
        @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
        async def well_known_oauth_server(request: Request) -> JSONResponse:
            return JSONResponse({
                "issuer": oauth_server.server_url,
                "authorization_endpoint": f"{oauth_server.server_url}/authorize",
                "token_endpoint": f"{oauth_server.server_url}/token",
                "registration_endpoint": f"{oauth_server.server_url}/register",
                "userinfo_endpoint": f"{oauth_server.server_url}/userinfo",
                "response_types_supported": ["code"],
                "response_modes_supported": ["query"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none"],
                "code_challenge_methods_supported": ["S256"],
                "scopes_supported": [
                    "openid", "profile", "email", "roles"
                ],
            })

        @mcp.custom_route("/register", methods=["POST"])
        async def register_client(request: Request) -> JSONResponse:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid_request"}, status_code=400)
            
            client_id = str(uuid.uuid4())
            
            client = RegisteredClient(
                client_id=client_id,
                client_name=body.get("client_name", "MCP Client"),
                redirect_uris=body.get("redirect_uris", []),
                grant_types=body.get("grant_types", ["authorization_code", "refresh_token"]),
                response_types=body.get("response_types", ["code"]),
                scope=body.get("scope"),
                token_endpoint_auth_method="none",
                created_at=datetime.now().timestamp()
            )
            
            oauth_server.registered_clients[client_id] = client
            
            return JSONResponse(client.model_dump(), status_code=201)

        @mcp.custom_route("/authorize", methods=["GET"])
        async def authorize(request: Request) -> RedirectResponse:
            # Get query parameters
            params = dict(request.query_params)
            
            # Remove our client_id and use Keycloak's client_id
            params.pop('client_id', None)
            params['client_id'] = keycloak_config.client_id
            
            # Build Keycloak auth URL
            auth_url = f"{keycloak_config.auth_endpoint}?{urlencode(params)}"
            
            return RedirectResponse(auth_url)

        @mcp.custom_route("/token", methods=["POST"])
        async def token(request: Request) -> JSONResponse:
            try:
                form = await request.form()
                grant_type = form.get("grant_type")
                
                if grant_type == "authorization_code":
                    return await _exchange_code_for_token(form)
                elif grant_type == "refresh_token":
                    return await _refresh_access_token(form)
                else:
                    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
            
            except Exception as e:
                logger.error(f"Token exchange error: {e}")
                return JSONResponse({"error": "invalid_request"}, status_code=400)

        @mcp.custom_route("/userinfo", methods=["GET"])
        async def userinfo(request: Request) -> JSONResponse:
            """Get user information from Keycloak"""
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse({"error": "invalid_token"}, status_code=401)
            
            try:
                response = await oauth_server.http_client.get(
                    keycloak_config.userinfo_endpoint,
                    headers={"Authorization": auth_header}
                )
                
                if response.status_code != 200:
                    return JSONResponse({"error": "invalid_token"}, status_code=401)
                
                userinfo = response.json()
                return JSONResponse(userinfo)
                
            except Exception as e:
                logger.error(f"Userinfo error: {e}")
                return JSONResponse({"error": "server_error"}, status_code=500)

        @mcp.custom_route("/health", methods=["GET"])
        async def health(request: Request):
            return JSONResponse({"status": "healthy", "timestamp": datetime.now().isoformat()})

        @mcp.custom_route("/debug", methods=["GET"])
        async def debug(request: Request):
            return JSONResponse({
                "registered_clients": len(oauth_server.registered_clients),
                "keycloak_config": {
                    "realm_url": keycloak_config.realm_url,
                    "client_id": keycloak_config.client_id,
                    "auth_endpoint": keycloak_config.auth_endpoint,
                    "token_endpoint": keycloak_config.token_endpoint
                },
                "server_url": server_url,
                "oauth_middleware_enabled": enable_oauth_middleware,
                "available_tools": list(mcp._tools.keys()) if hasattr(mcp, '_tools') else []
            })

        # Helper functions for token handling
        async def _exchange_code_for_token(form) -> JSONResponse:
            """Exchange authorization code for access token"""
            token_data = {
                "grant_type": "authorization_code",
                "code": form.get("code"),
                "redirect_uri": form.get("redirect_uri"),
                "client_id": keycloak_config.client_id,
                "client_secret": keycloak_config.client_secret,
            }
            
            # Add code_verifier for PKCE flow if present
            if form.get("code_verifier"):
                token_data["code_verifier"] = form.get("code_verifier")
            
            try:
                response = await oauth_server.http_client.post(
                    keycloak_config.token_endpoint,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Keycloak token exchange failed: {error_text}")
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)
                
                token_response = response.json()
                return JSONResponse(token_response)
                
            except Exception as e:
                logger.error(f"Token exchange error: {e}")
                return JSONResponse({"error": "server_error"}, status_code=500)

        async def _refresh_access_token(form) -> JSONResponse:
            """Refresh access token"""
            token_data = {
                "grant_type": "refresh_token",
                "refresh_token": form.get("refresh_token"),
                "client_id": keycloak_config.client_id,
                "client_secret": keycloak_config.client_secret,
            }
            
            try:
                response = await oauth_server.http_client.post(
                    keycloak_config.token_endpoint,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Keycloak token refresh failed: {error_text}")
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)
                
                token_response = response.json()
                return JSONResponse(token_response)
                
            except Exception as e:
                logger.error(f"Token refresh error: {e}")
                return JSONResponse({"error": "server_error"}, status_code=500)

        # Get the HTTP app from FastMCP
        app = mcp.http_app()
        
        # Add authentication middleware only if enabled
        if enable_oauth_middleware:
            logger.info("OAuth middleware enabled")
            # app.add_middleware(AuthAndWWWMiddleware, oauth_server=oauth_server, server_url=server_url)
        else:
            logger.info("OAuth middleware disabled")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Add shutdown handler
        @app.on_event("shutdown")
        async def shutdown_event():
            logger.info("Shutting down server...")
            await oauth_server.close()
            if hasattr(rag_manager, 'close'):
                await rag_manager.close()
            logger.info("Server shutdown complete")
        
        # Start server
        logger.info(f"Keycloak realm URL: {keycloak_realm_url}")
        logger.info(f"Server URL: {server_url}")
        logger.info(f"OAuth middleware enabled: {enable_oauth_middleware}")
        logger.info(f"Config file: {config_path}")
        
        config_uvicorn = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config_uvicorn)
        await server.serve()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main entry point"""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()