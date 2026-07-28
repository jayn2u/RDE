import unittest

import torch

from utils.metrics import compute_retrieval_metrics


class RetrievalMetricsTest(unittest.TestCase):
    def test_combined_branch_populates_primary_metrics(self):
        identities = torch.arange(10)
        bge = torch.eye(10) * 10.0
        tse = torch.fliplr(torch.eye(10)) * 20.0 - torch.eye(10) * 20.0
        combined = (bge + tse) / 2.0

        metrics = compute_retrieval_metrics(
            {"BGE": bge, "TSE": tse, "BGE+TSE": combined},
            query_ids=identities,
            gallery_ids=identities,
            include_i2t=True,
        )

        self.assertEqual(metrics["bge_t2i_R1"], 100.0)
        self.assertEqual(metrics["t2i_R1"], 0.0)
        self.assertEqual(metrics["i2t_R1"], 0.0)
        self.assertIn("tse_t2i_mAP", metrics)
        self.assertIn("bge_i2t_mINP", metrics)

    def test_component_keys_are_not_duplicated_for_combined_branch(self):
        identities = torch.arange(10)
        similarity = torch.eye(10)

        metrics = compute_retrieval_metrics(
            {
                "BGE": similarity,
                "TSE": similarity,
                "BGE+TSE": similarity,
            },
            query_ids=identities,
            gallery_ids=identities,
            include_i2t=False,
        )

        self.assertNotIn("bge+tse_t2i_R1", metrics)
        self.assertEqual(metrics["t2i_R10"], 100.0)
        self.assertNotIn("i2t_R1", metrics)


if __name__ == "__main__":
    unittest.main()
