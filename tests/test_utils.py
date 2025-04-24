import unittest
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from unillm.llm_models import base642img, img2base64

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        # Create a simple test image (10x10 red square)
        self.test_img = Image.new('RGB', (10, 10), color='red')
        
    def test_img2base64(self):
        """Test conversion from PIL Image to base64 string."""
        base64_str = img2base64(self.test_img)
        
        # Assert that we get a non-empty string
        self.assertIsInstance(base64_str, str)
        self.assertTrue(len(base64_str) > 0)
        
        # Verify it can be decoded as base64
        try:
            decoded = base64.b64decode(base64_str)
            self.assertTrue(len(decoded) > 0)
        except Exception as e:
            self.fail(f"Failed to decode base64 string: {e}")
    
    def test_base642img(self):
        """Test conversion from base64 string to PIL Image."""
        # First convert image to base64
        buffered = BytesIO()
        self.test_img.save(buffered, format="JPEG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Now convert back to image
        img = base642img(base64_str)
        
        # Assert that we get a PIL image back
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, 'RGB')  # JPEG format should be RGB
    
    def test_round_trip_conversion(self):
        """Test that converting to base64 and back preserves the image."""
        # Convert to base64
        base64_str = img2base64(self.test_img)
        
        # Convert back to image
        img = base642img(base64_str)
        
        # Compare the images (they won't be identical due to JPEG compression, 
        # but should be very similar)
        # Convert both images to numpy arrays for comparison
        original_array = np.array(self.test_img)
        converted_array = np.array(img)
        
        # For a simple red image, the main RGB values should still be predominantly red
        # after compression
        self.assertTrue(np.mean(converted_array[:,:,0]) > 200)  # Red channel should be high
        
        # Image dimensions should be preserved
        self.assertEqual(img.size, self.test_img.size)

if __name__ == "__main__":
    unittest.main()