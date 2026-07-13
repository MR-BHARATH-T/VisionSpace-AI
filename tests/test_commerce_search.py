import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure the root path is in the sys path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from agents.commerce_agent import CommerceAgent

class TestCommerceSearch(unittest.TestCase):
    def setUp(self):
        # Instantiate agent
        self.agent = CommerceAgent(provider="huggingface")
        # Ensure SERPAPI key is present to trigger SerpAPI flow in tests
        os.environ["SERP_API_KEY"] = "mock_key"

    @patch("urllib.request.urlopen")
    def test_scenario_a_branded_success(self, mock_urlopen):
        # Scenario A: Branded search succeeds
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"shopping_results": [{"title": "IKEA Table", "price": "$100", "brand": "IKEA", "thumbnail": "https://example.com/img.png"}]}'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        results = self.agent.search_products(categories=["study table"], preferred_brands=["IKEA"])
        
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["brand"], "IKEA")

    @patch("urllib.request.urlopen")
    def test_scenario_b_branded_retry_success(self, mock_urlopen):
        # Scenario B: First branded search times out, retry succeeds
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"shopping_results": [{"title": "IKEA Table", "price": "$100", "brand": "IKEA", "thumbnail": "https://example.com/img.png"}]}'
        mock_response.__enter__.return_value = mock_response
        
        # Raise timeout exception on first call, succeed on second
        mock_urlopen.side_effect = [TimeoutError("The read operation timed out"), mock_response]
        
        results = self.agent.search_products(categories=["study table"], preferred_brands=["IKEA"])
        
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["brand"], "IKEA")

    @patch("urllib.request.urlopen")
    def test_scenario_c_branded_fails_generic_succeeds(self, mock_urlopen):
        # Scenario C: All branded searches fail, generic fallback succeeds
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"shopping_results": [{"title": "Generic Table", "price": "$80", "brand": "Generic", "thumbnail": "https://example.com/img.png"}]}'
        mock_response.__enter__.return_value = mock_response
        
        # Fails twice for branded search, succeeds for generic fallback search
        mock_urlopen.side_effect = [
            TimeoutError("The read operation timed out"), # branded attempt 1
            TimeoutError("The read operation timed out"), # branded attempt 2
            mock_response                                 # generic fallback
        ]
        
        results = self.agent.search_products(categories=["study table"], preferred_brands=["IKEA"])
        
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["brand"], "Generic")

    @patch("urllib.request.urlopen")
    def test_scenario_d_all_serp_fails_local_catalog_succeeds(self, mock_urlopen):
        # Scenario D: Both branded and generic searches fail - falls back to local catalog
        mock_urlopen.side_effect = TimeoutError("The read operation timed out")
        
        results = self.agent.search_products(categories=["study table"], preferred_brands=["IKEA"])
        
        # Verify that it falls back to local catalog and does not return empty/crash
        self.assertTrue(len(results) > 0)

    @patch("urllib.request.urlopen")
    def test_scenario_e_partial_item_success(self, mock_urlopen):
        # Scenario E: One required item fails completely (returns mock or generic), other required items succeed
        mock_response_1 = MagicMock()
        mock_response_1.read.return_value = b'{"shopping_results": [{"title": "IKEA Table", "price": "$120", "brand": "IKEA", "thumbnail": "https://example.com/img.png"}]}'
        mock_response_1.__enter__.return_value = mock_response_1
        
        # First category (table) succeeds, second category (cooler) times out on everything and returns mock
        mock_urlopen.side_effect = [
            mock_response_1,                              # branded search table
            TimeoutError("The read operation timed out"), # branded search cooler attempt 1
            TimeoutError("The read operation timed out"), # branded search cooler attempt 2
            TimeoutError("The read operation timed out"), # generic search cooler attempt 1
            TimeoutError("The read operation timed out"), # generic search cooler attempt 2
        ]
        
        results = self.agent.search_products(categories=["study table", "air cooler"], preferred_brands=["IKEA"])
        
        # Should have candidates from both categories (table from SerpAPI, cooler from mock catalog fallback)
        categories_returned = [r["category"].lower() for r in results]
        self.assertTrue(any("table" in c or "desk" in c for c in categories_returned))
        self.assertTrue(any("cooler" in c or "appliance" in c for c in categories_returned))

if __name__ == "__main__":
    unittest.main()
