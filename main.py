from dataclasses import dataclass
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime, timedelta
from pathlib import Path

@dataclass
class ZohoConfig:
    """Configuration data class for Zoho API credentials and settings"""
    client_id: str
    client_secret: str
    refresh_token: str
    department_id: str

class ZohoAPIError(Exception):
    """Custom exception for Zoho API related errors"""
    pass

class ZohoSupportBot:
    def __init__(self, config_path: str = 'zoho_config.json'):
        """
        Initialize the Zoho support bot with configuration
        
        Args:
            config_path: Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        self._setup_logging()
        self.config = self._load_config(config_path)
        self.access_token: Optional[str] = None
        self.model: Optional[BaseEstimator] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        
    def _setup_logging(self) -> None:
        """Configure logging with rotating file handler"""
        logging.basicConfig(
            filename='zoho_support_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _load_config(self, config_path: str) -> ZohoConfig:
        """
        Load and validate configuration from JSON file
        
        Returns:
            ZohoConfig object containing validated configuration
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                return ZohoConfig(**config_data)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in configuration file: {config_path}")
            raise

    def _make_api_request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make authenticated request to Zoho API with retry logic
        
        Args:
            method: HTTP method ('GET' or 'POST')
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            JSON response from API
            
        Raises:
            ZohoAPIError: If API request fails after retries
        """
        MAX_RETRIES = 3
        base_url = "https://desk.zoho.com/api/v1"
        
        for attempt in range(MAX_RETRIES):
            try:
                if not self.access_token:
                    self.get_access_token()
                
                headers = {
                    'Authorization': f'Zoho-oauthtoken {self.access_token}',
                    **kwargs.get('headers', {})
                }
                
                response = requests.request(
                    method,
                    f"{base_url}/{endpoint.lstrip('/')}",
                    headers=headers,
                    **{k: v for k, v in kwargs.items() if k != 'headers'}
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise ZohoAPIError(f"API request failed: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
                self.access_token = None  # Force token refresh on retry

    def get_access_token(self) -> str:
        """
        Get Zoho OAuth access token using refresh token
        
        Returns:
            Valid access token
            
        Raises:
            ZohoAPIError: If token refresh fails
        """
        try:
            response = requests.post(
                "https://accounts.zoho.com/oauth/v2/token",
                data={
                    'refresh_token': self.config.refresh_token,
                    'client_id': self.config.client_id,
                    'client_secret': self.config.client_secret,
                    'grant_type': 'refresh_token'
                }
            )
            response.raise_for_status()
            self.access_token = response.json()['access_token']
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            raise ZohoAPIError(f"Failed to refresh access token: {str(e)}")

    def fetch_historical_tickets(self, months_back: int = 6) -> List[Dict[str, Any]]:
        """
        Fetch historical tickets from Zoho Desk with pagination
        
        Args:
            months_back: Number of months of history to fetch
            
        Returns:
            List of ticket data dictionaries
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months_back)
        
        tickets = []
        from_index = 0
        limit = 100
        
        while True:
            response = self._make_api_request(
                'GET',
                'tickets',
                params={
                    'departmentId': self.config.department_id,
                    'limit': limit,
                    'from': from_index,
                    'modifiedTimeRange': f"{start_date.isoformat()},{end_date.isoformat()}"
                }
            )
            
            if not response['data']:
                break
                
            tickets.extend(response['data'])
            from_index += limit
            
            if from_index >= response['count']:
                break
                
            time.sleep(1)  # Rate limiting
        
        return tickets

    def prepare_training_data(self, tickets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare training data from tickets, filtering for resolved tickets with responses
        
        Args:
            tickets: List of ticket data from API
            
        Returns:
            DataFrame with query, response, and category columns
        """
        data = []
        for ticket in tickets:
            if ticket['status'] == 'Closed' and ticket['threadCount'] > 1:
                ticket_detail = self.get_ticket_details(ticket['id'])
                if ticket_detail and ticket_detail.get('threadsContent'):
                    data.append({
                        'query': ticket_detail['threadsContent'][0]['content'],
                        'response': ticket_detail['threadsContent'][1]['content'],
                        'category': ticket.get('category', 'General')
                    })
        
        df = pd.DataFrame(data)
        logging.info(f"Prepared training data with {len(df)} samples")
        return df

    def train_model(self, data: pd.DataFrame) -> None:
        """
        Train and save classification model
        
        Args:
            data: Training data DataFrame
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95
        )
        X = self.vectorizer.fit_transform(data['query'])
        y = pd.get_dummies(data['category'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced'
            )
        )
        self.model.fit(X_train, y_train)
        
        self._save_model()
        
        accuracy = self.model.score(X_test, y_test)
        logging.info(f"Model trained with accuracy: {accuracy:.2f}")

    def _save_model(self) -> None:
        """Save trained model and vectorizer to disk"""
        model_path = Path('support_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump((self.model, self.vectorizer), f)
        logging.info(f"Saved model to {model_path}")

    def run(self, check_interval: int = 300) -> None:
        """
        Main bot loop to process new tickets
        
        Args:
            check_interval: Seconds between checking for new tickets
        """
        self._load_or_train_model()
        
        while True:
            try:
                self.process_new_tickets()
                time.sleep(check_interval)
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                continue

    def _load_or_train_model(self) -> None:
        """Load existing model or train new one if not found"""
        try:
            with open('support_model.pkl', 'rb') as f:
                self.model, self.vectorizer = pickle.load(f)
            logging.info("Loaded existing model")
        except FileNotFoundError:
            logging.info("Training new model...")
            tickets = self.fetch_historical_tickets()
            data = self.prepare_training_data(tickets)
            self.train_model(data)