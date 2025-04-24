import unittest
from unillm.llm_models import SamplingParams

class TestSamplingParams(unittest.TestCase):
    
    def test_default_params(self):
        """Test that default parameters are set correctly."""
        params = SamplingParams()
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.max_tokens, 2048)
        self.assertEqual(params.top_p, 0.9)
        self.assertEqual(params.n, 1)
        self.assertIsNone(params.stop)
    
    def test_custom_params(self):
        """Test that custom parameters are set correctly."""
        params = SamplingParams(
            temperature=0.5,
            max_tokens=1024,
            top_p=0.8,
            n=3,
            stop=["END"]
        )
        self.assertEqual(params.temperature, 0.5)
        self.assertEqual(params.max_tokens, 1024)
        self.assertEqual(params.top_p, 0.8)
        self.assertEqual(params.n, 3)
        self.assertEqual(params.stop, ["END"])

if __name__ == "__main__":
    unittest.main()