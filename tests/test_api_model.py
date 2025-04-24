import unittest
from unittest.mock import patch, MagicMock
import os
import json
from io import BytesIO
from unillm.llm_models import APIModel, SamplingParams

class TestAPIModel(unittest.TestCase):
    
    def setUp(self):
        # Set environment variables for testing
        os.environ["OAI_API_KEY"] = "test-openai-key"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        os.environ["GOOGLE_API_KEY"] = "test-google-key"
        os.environ["GOOGLE_GENAI_API_KEY"] = "test-google-genai-key"

        # Default sampling params for tests
        self.sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=1000, 
            top_p=1.0,
            n=1
        )
        
        # Example prompt
        self.prompt = [
            {
                "role": "user", 
                "content": "Hello, how are you?"
            }
        ]
    
    def tearDown(self):
        # Clean up environment variables
        for key in ["OAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
    
    @patch('unillm.llm_models.OpenAI')
    def test_openai_initialization(self, mock_openai):
        """Test that the OpenAI client is initialized correctly."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        model = APIModel("gpt-4o", provider="openai")
        
        # Check that OpenAI was initialized with the correct API key
        mock_openai.assert_called_once_with(api_key="test-openai-key", base_url=None)
        self.assertEqual(model.model_name, "gpt-4o")
        self.assertEqual(model.provider, "openai")
        
    @patch('unillm.llm_models.OpenAI')
    def test_google_initialization(self, mock_openai):
        """Test that the Google (OpenAI-compatible) client is initialized correctly."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        model = APIModel("gemini-pro", provider="google")
        
        # Check that OpenAI was initialized with the correct API key and base URL
        mock_openai.assert_called_once_with(
            api_key="test-google-key", 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.assertEqual(model.model_name, "gemini-pro")
        self.assertEqual(model.provider, "google")
        
    @patch('unillm.llm_models.anthropic.Anthropic')
    def test_anthropic_initialization(self, mock_anthropic):
        """Test that the Anthropic client is initialized correctly."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        model = APIModel("claude-3-opus", provider="anthropic")
        
        # Check that Anthropic was initialized with the correct API key
        mock_anthropic.assert_called_once_with(api_key="test-anthropic-key")
        self.assertEqual(model.model_name, "claude-3-opus")
        self.assertEqual(model.provider, "anthropic")
        
    @patch('unillm.llm_models.genai.Client')
    def test_google_genai_initialization(self, mock_genai):
        """Test that the Google GenAI client is initialized correctly."""
        mock_client = MagicMock()
        mock_genai.return_value = mock_client
        
        model = APIModel("gemini-1.0-pro-codeinterpreter", provider="google-genai")
        
        # Check that Google GenAI was initialized with the correct API key
        mock_genai.assert_called_once_with(api_key="test-google-genai-key")
        self.assertEqual(model.model_name, "gemini-1.0-pro-codeinterpreter")
        self.assertEqual(model.provider, "google-genai")
        
    @patch('unillm.llm_models.boto3.client')
    def test_bedrock_initialization(self, mock_boto3_client):
        """Test that the AWS Bedrock client is initialized correctly."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        model = APIModel("anthropic.claude-3-sonnet-20240229-v1:0", provider="bedrock")
        
        # Check that boto3 client was initialized with the correct parameters
        mock_boto3_client.assert_called_once()
        self.assertEqual(mock_boto3_client.call_args[0][0], "bedrock-runtime")
        self.assertEqual(mock_boto3_client.call_args[1]["region_name"], "us-east-1")
        self.assertEqual(model.model_name, "anthropic.claude-3-sonnet-20240229-v1:0")
        self.assertEqual(model.provider, "bedrock")
        
    @patch('unillm.llm_models.OpenAI')
    def test_openai_chat_completion(self, mock_openai):
        """Test that OpenAI chat completion works correctly."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello, I'm an AI assistant."))]
        mock_response.usage = MagicMock(completion_tokens=100)
        
        # Set up the mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Initialize the model and call chat
        model = APIModel("gpt-4o", provider="openai")
        result = model.chat(self.prompt, self.sampling_params, use_tqdm=False)
        
        # Check that the response was processed correctly
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].outputs), 1)
        self.assertEqual(result[0].outputs[0].text, "Hello, I'm an AI assistant.")
        
        # Verify the API was called with the expected parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o")
        self.assertEqual(call_args["messages"], self.prompt)
        self.assertEqual(call_args["temperature"], 0.0)
        
    @patch('unillm.llm_models.OpenAI')  
    def test_google_chat_completion(self, mock_openai):
        """Test that Google (OpenAI-compatible) chat completion works correctly."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello, I'm Gemini via OpenAI API!"))]
        mock_response.usage = MagicMock(completion_tokens=100)
        
        # Set up the mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Initialize the model and call chat
        model = APIModel("gemini-pro", provider="google")
        result = model.chat(self.prompt, self.sampling_params, use_tqdm=False)
        
        # Check that the response was processed correctly
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].outputs), 1)
        self.assertEqual(result[0].outputs[0].text, "Hello, I'm Gemini via OpenAI API!")
        
        # Verify the API was called with the expected parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gemini-pro")
        self.assertEqual(call_args["messages"], self.prompt)
        
    @patch('unillm.llm_models.anthropic.Anthropic')
    def test_anthropic_chat_completion(self, mock_anthropic):
        """Test that Anthropic chat completion works correctly."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello, I'm Claude!")]
        
        # Set up the mock client
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Initialize the model and call chat
        model = APIModel("claude-3-opus", provider="anthropic")
        result = model.chat(self.prompt, self.sampling_params, use_tqdm=False)
        
        # Check that the response was processed correctly
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].outputs), 1)
        self.assertEqual(result[0].outputs[0].text, "Hello, I'm Claude!")
        
        # Verify the API was called with the expected parameters
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["model"], "claude-3-opus")
        self.assertEqual(call_args["messages"], self.prompt)
        self.assertEqual(call_args["temperature"], 0.0)
        
    @patch('unillm.llm_models.boto3.client')
    def test_bedrock_chat_completion(self, mock_boto3_client):
        """Test that AWS Bedrock chat completion works correctly."""
        # Create a mock response
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = json.dumps({
            'content': [{'text': "Hello, I'm Claude on Bedrock!"}]
        })
        
        # Set up the mock client
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = mock_response
        mock_boto3_client.return_value = mock_client
        
        # Initialize the model and call chat
        model = APIModel("anthropic.claude-3-sonnet-20240229-v1:0", provider="bedrock")
        result = model.chat(self.prompt, self.sampling_params, use_tqdm=False)
        
        # Check that the response was processed correctly
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].outputs), 1)
        self.assertEqual(result[0].outputs[0].text, "Hello, I'm Claude on Bedrock!")
        
        # Verify the API was called with the expected parameters
        mock_client.invoke_model.assert_called_once()
        
    @patch('unillm.llm_models.genai.Client')
    def test_google_genai_chat_completion(self, mock_genai_client):
        """Test that Google GenAI chat completion works correctly."""
        # Create a mock model and response
        mock_model = MagicMock()
        mock_response = MagicMock()
        
        # Configure the response
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Hello, I'm Gemini!"
        mock_part.executable_code = None
        mock_part.code_execution_result = None
        
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None
        
        # Setup the client mock chain
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        mock_client = MagicMock()
        mock_client.models = mock_models
        mock_genai_client.return_value = mock_client
        
        # Initialize the model and call chat
        model = APIModel("gemini-1.0-pro-codeinterpreter", provider="google-genai")
        result = model.chat(self.prompt, self.sampling_params, use_tqdm=False)
        
        # Check that the response was processed correctly
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].outputs), 1)
        self.assertEqual(result[0].outputs[0].text, "Hello, I'm Gemini!")
        
        # Verify the API was called with the expected parameters
        mock_models.generate_content.assert_called_once()

if __name__ == "__main__":
    unittest.main()