import unittest
from unillm.llm_models import UniLLM
from abc import ABC

class TestUniLLM(unittest.TestCase):
    
    def test_unillm_is_abstract(self):
        """Test that UniLLM is an abstract base class."""
        self.assertTrue(issubclass(UniLLM, ABC))
        
    def test_abstract_methods(self):
        """Test that abstract methods are defined correctly."""
        # Check if the abstract methods are defined in the class
        abstract_methods = UniLLM.__abstractmethods__
        self.assertIn('__init__', abstract_methods)
        self.assertIn('chat', abstract_methods)
        
    def test_instantiation_not_allowed(self):
        """Test that direct instantiation of UniLLM is not allowed."""
        with self.assertRaises(TypeError):
            UniLLM("test-model")

if __name__ == "__main__":
    unittest.main()