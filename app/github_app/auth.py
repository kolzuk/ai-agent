"""GitHub App authentication with JWT and installation tokens."""

import time
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

import jwt
import httpx
from cryptography.hazmat.primitives import serialization

from ..config import settings

logger = logging.getLogger(__name__)


class GitHubAppAuth:
    """GitHub App authentication manager."""
    
    def __init__(self):
        self.app_id = settings.github_app_id
        self._private_key: Optional[str] = None
        self._installation_tokens: Dict[int, Dict] = {}  # Cache for installation tokens
    
    @property
    def private_key(self) -> str:
        """Lazy load private key."""
        if self._private_key is None:
            self._private_key = self._load_private_key()
        return self._private_key
    
    def _load_private_key(self) -> str:
        """Load and validate private key."""
        try:
            private_key_content = settings.get_private_key()
            
            # Validate the private key by trying to load it
            serialization.load_pem_private_key(
                private_key_content.encode(),
                password=None
            )
            
            return private_key_content
        except Exception as e:
            logger.error(f"Failed to load GitHub App private key: {e}")
            raise ValueError(f"Invalid GitHub App private key: {e}")
    
    def generate_jwt(self) -> str:
        """Generate JWT token for GitHub App authentication."""
        now = int(time.time())
        
        payload = {
            "iat": now - 60,  # Issued at time (60 seconds ago to account for clock skew)
            "exp": now + 600,  # Expires in 10 minutes
            "iss": self.app_id  # Issuer (GitHub App ID)
        }
        
        try:
            token = jwt.encode(
                payload,
                self.private_key,
                algorithm="RS256"
            )
            return token
        except Exception as e:
            logger.error(f"Failed to generate JWT: {e}")
            raise ValueError(f"Failed to generate JWT: {e}")
    
    async def get_installation_token(self, installation_id: int) -> str:
        """Get installation access token for a specific installation."""
        # Check if we have a cached token that's still valid
        if installation_id in self._installation_tokens:
            token_data = self._installation_tokens[installation_id]
            expires_at = datetime.fromisoformat(token_data["expires_at"].replace("Z", "+00:00"))
            
            # If token expires in more than 5 minutes, use cached token
            if expires_at > datetime.now().astimezone() + timedelta(minutes=5):
                return token_data["token"]
        
        # Generate new installation token
        jwt_token = self.generate_jwt()
        
        url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Coding-Agent/1.0"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers)
                response.raise_for_status()
                
                token_data = response.json()
                
                # Cache the token
                self._installation_tokens[installation_id] = token_data
                
                logger.info(f"Generated new installation token for installation {installation_id}")
                return token_data["token"]
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get installation token: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Failed to get installation token: {e}")
        except Exception as e:
            logger.error(f"Error getting installation token: {e}")
            raise ValueError(f"Error getting installation token: {e}")
    
    async def get_installation_id(self, owner: str, repo: str) -> Optional[int]:
        """Get installation ID for a repository."""
        jwt_token = self.generate_jwt()
        
        url = f"https://api.github.com/repos/{owner}/{repo}/installation"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Coding-Agent/1.0"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 404:
                    logger.warning(f"GitHub App not installed on {owner}/{repo}")
                    return None
                
                response.raise_for_status()
                installation_data = response.json()
                return installation_data["id"]
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get installation ID: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Error getting installation ID: {e}")
            return None
    
    async def get_authenticated_client(self, installation_id: int) -> httpx.AsyncClient:
        """Get authenticated HTTP client for GitHub API."""
        token = await self.get_installation_token(installation_id)
        
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Coding-Agent/1.0"
        }
        
        return httpx.AsyncClient(headers=headers)
    
    def clear_token_cache(self, installation_id: Optional[int] = None) -> None:
        """Clear cached installation tokens."""
        if installation_id:
            self._installation_tokens.pop(installation_id, None)
        else:
            self._installation_tokens.clear()
        logger.info(f"Cleared token cache for installation {installation_id or 'all'}")


# Global GitHub App auth instance
github_app_auth = GitHubAppAuth()