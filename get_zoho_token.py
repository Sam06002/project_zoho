import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
from urllib.parse import urlparse, parse_qs

class AuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle the callback from Zoho"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Parse the authorization code from the callback URL
        query_components = parse_qs(urlparse(self.path).query)
        
        if 'code' in query_components:
            # Store the code in the class
            AuthHandler.authorization_code = query_components['code'][0]
            
            # Display success message
            self.wfile.write(b"Authorization successful! You can close this window.")
        else:
            self.wfile.write(b"Authorization failed! Please try again.")
        
        # Signal the server to stop
        self.server.stopped = True

def get_refresh_token(client_id: str, client_secret: str, port: int = 8080):
    """
    Get refresh token from Zoho OAuth
    
    Args:
        client_id: Your Zoho client ID
        client_secret: Your Zoho client secret
        port: Port to run local server on (default 8080)
    """
    
    # Step 1: Get authorization code
    scope = "ZohoSupport.tickets.ALL"
    redirect_uri = f"http://localhost:{8080}"
    
    auth_url = (
        "https://accounts.zoho.com/oauth/v2/auth?"
        f"scope={scope}&"
        f"client_id={client_id}&"
        "response_type=code&"
        f"redirect_uri={redirect_uri}&"
        "access_type=offline"
    )
    
    # Open browser for authentication
    print(f"Opening browser for authentication...")
    webbrowser.open(auth_url)
    
    # Start local server to receive callback
    server = HTTPServer(('localhost', port), AuthHandler)
    server.stopped = False
    
    print(f"Waiting for authentication callback on port {port}...")
    while not server.stopped:
        server.handle_request()
    
    # Get authorization code from handler
    authorization_code = getattr(AuthHandler, 'authorization_code', None)
    
    if not authorization_code:
        raise Exception("Failed to get authorization code")
    
    # Step 2: Exchange authorization code for refresh token
    token_url = "https://accounts.zoho.com/oauth/v2/token"
    data = {
        'code': authorization_code,
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code'
    }
    
    response = requests.post(token_url, data=data)
    if response.status_code != 200:
        raise Exception(f"Failed to get refresh token: {response.text}")
    
    token_data = response.json()
    refresh_token = token_data.get('refresh_token')
    
    if not refresh_token:
        raise Exception("No refresh token in response")
    
    return refresh_token

if __name__ == "__main__":
    # Replace these with your actual client ID and secret
    CLIENT_ID = "1000.28YFTM8IDT1GLTXYUS3ETIOW7IVHVK"
    CLIENT_SECRET = "19622d3904eceb3d5ed0c308c309b03a29ef47e556"
    
    try:
        refresh_token = get_refresh_token(CLIENT_ID, CLIENT_SECRET)
        print("\nYour refresh token:", refresh_token)
        print("\nAdd this token to your zoho_config.json file")
        
    except Exception as e:
        print(f"Error: {str(e)}")