#!/usr/bin/env python3
"""Example Python client for Companies House MCP Server with IAM authentication.

Install dependencies:
    pip install google-auth google-auth-oauthlib google-auth-httplib2 requests

Usage with Service Account Key File:
    export SERVICE_URL="https://your-service-url.run.app"
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
    python python_client_iam_mcp.py
"""

import base64
import json
import os
import sys
import time
from typing import Any, Dict

try:
    import google.auth
    import requests
    from google.auth import exceptions as google_auth_exceptions
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 requests")
    sys.exit(1)


class IAMAuthenticatedMCPClient:
    """Client for making authenticated requests to IAM-protected MCP server."""
    
    def __init__(self, service_url: str):
        # Remove trailing slash to avoid double slashes in URLs
        self.service_url = service_url.rstrip('/')
        self._id_token = None

    def _token_is_expired(self) -> bool:
        """Return True if the cached token is missing or expires within 5 minutes."""
        if not self._id_token:
            return True
        try:
            parts = self._id_token.split('.')
            if len(parts) < 2:
                return True
            padded = parts[1] + '=' * (-len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            return time.time() > payload.get('exp', 0) - 300
        except Exception:
            return True

    def _get_identity_token(self) -> str:
        """Get or refresh identity token for the service using service account credentials.

        Supports two credential sources (checked in order):
        1. GOOGLE_APPLICATION_CREDENTIALS_JSON — raw JSON string of the service account key.
           Set this in cloud environments (e.g. LangGraph Cloud) where file paths are unavailable.
        2. GOOGLE_APPLICATION_CREDENTIALS — path to a service account key file (local dev default).
        """
        if not self._token_is_expired():
            return self._id_token

        creds_json_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not creds_json_str and not creds_path:
            raise Exception(
                "No GCP credentials found. Set either:\n"
                "  GOOGLE_APPLICATION_CREDENTIALS_JSON — service account key JSON string (cloud deployments)\n"
                "  GOOGLE_APPLICATION_CREDENTIALS      — path to a service account key file (local dev)"
            )

        try:
            # Use IDTokenCredentials to sign the ID token directly with the private key.
            # This bypasses fetch_id_token() which relies on ADC/metadata server and
            # fails in non-GCP cloud environments like LangGraph Cloud.
            if creds_json_str:
                service_account_info = json.loads(creds_json_str)
                id_token_credentials = service_account.IDTokenCredentials.from_service_account_info(
                    service_account_info,
                    target_audience=self.service_url,
                )
            else:
                if not os.path.isfile(creds_path):
                    raise Exception(
                        f"Service account key file not found: {creds_path}\n"
                        "Please verify the GOOGLE_APPLICATION_CREDENTIALS path is correct."
                    )
                id_token_credentials = service_account.IDTokenCredentials.from_service_account_file(
                    creds_path,
                    target_audience=self.service_url,
                )

            request = Request()
            id_token_credentials.refresh(request)
            self._id_token = id_token_credentials.token

            if not self._id_token:
                raise Exception("Failed to obtain ID token - token is None")
            
            return self._id_token
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a credentials error
            is_credential_error = (
                isinstance(e, (google_auth_exceptions.DefaultCredentialsError, google_auth_exceptions.GoogleAuthError)) or
                "credentials" in error_msg.lower() or
                "invalid" in error_msg.lower()
            )
            
            if is_credential_error:
                raise Exception(
                    f"Authentication failed: {error_msg}\n\n"
                    "Please ensure:\n"
                    "  1. GOOGLE_APPLICATION_CREDENTIALS is set to a valid service account key file path\n"
                    "  2. The service account key file is valid JSON\n"
                    "  3. The service account has 'roles/run.invoker' permission on the Cloud Run service\n\n"
                    "To create a service account key file:\n"
                    "  $ gcloud iam service-accounts keys create key.json \\\n"
                    "      --iam-account=SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com\n\n"
                    "To grant the service account access to the Cloud Run service:\n"
                    "  $ gcloud run services add-iam-policy-binding SERVICE_NAME \\\n"
                    "      --region=REGION \\\n"
                    "      --member=\"serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com\" \\\n"
                    "      --role=\"roles/run.invoker\""
                )
            else:
                # Some other error - re-raise it
                raise
    
    def _make_request(self, method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Make an authenticated JSON-RPC request."""
        token = self._get_identity_token()
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.service_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Token might be expired, refresh and retry once
                self._id_token = None
                token = self._get_identity_token()
                headers["Authorization"] = f"Bearer {token}"
                response = requests.post(
                    self.service_url,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            raise
    
    def list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        return self._make_request("tools/list")
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool."""
        return self._make_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        token = self._get_identity_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        # Ensure no double slashes in URL
        health_url = self.service_url.rstrip('/') + '/health'
        try:
            response = requests.get(health_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 401:
                # Get response body for more details
                error_detail = ""
                try:
                    error_body = e.response.json() if e.response.content else {}
                    error_detail = error_body.get("error", {}).get("message", "") or error_body.get("message", "")
                except:
                    error_detail = e.response.text[:200] if e.response.text else ""
                
                # Get service account email from credentials
                creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                service_account_email = ""
                if creds_path and os.path.isfile(creds_path):
                    try:
                        with open(creds_path) as f:
                            creds_json = json.load(f)
                            service_account_email = creds_json.get("client_email", "")
                    except:
                        pass
                
                error_msg = "401 Unauthorized: The service account does not have permission to access the Cloud Run service.\n\n"
                if error_detail:
                    error_msg += f"Error details: {error_detail}\n\n"
                if service_account_email:
                    error_msg += f"The service account '{service_account_email}' needs 'roles/run.invoker' permission.\n\n"
                error_msg += "Grant the 'roles/run.invoker' role to the service account:\n"
                error_msg += "  $ gcloud run services add-iam-policy-binding SERVICE_NAME \\\n"
                error_msg += "      --region=REGION \\\n"
                if service_account_email:
                    error_msg += f"      --member=\"serviceAccount:{service_account_email}\" \\\n"
                else:
                    error_msg += "      --member=\"serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com\" \\\n"
                error_msg += "      --role=\"roles/run.invoker\""
                
                raise Exception(error_msg)
            raise


def main():
    """Example usage of the IAM-authenticated client."""
    service_url = os.getenv("SERVICE_URL")
    if not service_url:
        print("Error: SERVICE_URL environment variable not set")
        print("Set it with: export SERVICE_URL='https://your-service-url.run.app'")
        sys.exit(1)
    
    # Check for service account key file
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        print("Set it with: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account-key.json'")
        sys.exit(1)
    
    if not os.path.isfile(credentials_path):
        print(f"Error: Service account key file not found: {credentials_path}")
        print("Please set GOOGLE_APPLICATION_CREDENTIALS to a valid service account key file path.")
        sys.exit(1)
    
    print(f"Using service account key file: {credentials_path}")
    print()
    
    client = IAMAuthenticatedMCPClient(service_url)
    
    try:
        # Health check
        print("Checking service health...")
        health = client.health_check()
        print(f"Health: {json.dumps(health, indent=2)}\n")
        
        # List available tools
        print("Fetching available tools...")
        tools_response = client.list_tools()
        tools = tools_response.get("result", {}).get("tools", [])
        print(f"Found {len(tools)} tools\n")
        
        # Show first few tools
        print("Sample tools:")
        for tool in tools[:5]:
            print(f"  - {tool['name']}: {tool['description'][:60]}...")
        
        # Example: Search for companies
        print("\nSearching for companies...")
        search_result = client.call_tool("search_companies", {
            "query": "Goodmans Consulting",
            "items_per_page": 5
        })
        
        if "error" in search_result:
            print(f"Error: {search_result['error']}")
        else:
            result_content = search_result.get("result", {}).get("content", [])
            if result_content:
                print("Search results:")
                print(json.dumps(json.loads(result_content[0]["text"]), indent=2))
        
    except Exception as e:
        # Print the full error message (which may contain helpful instructions)
        error_msg = str(e)
        if "401 Unauthorized" in error_msg or "roles/run.invoker" in error_msg:
            # This is a permission error with helpful instructions
            print(error_msg)
        else:
            # Generic error
            print(f"Error: {error_msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

