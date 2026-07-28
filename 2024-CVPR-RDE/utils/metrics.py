from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
 
from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]


def _direction_metrics(similarity, query_ids, gallery_ids, prefix):
    cmc, mean_ap, mean_inp, _ = rank(
        similarity=similarity,
        q_pids=query_ids,
        g_pids=gallery_ids,
        max_rank=10,
        get_mAP=True,
    )
    return {
        f"{prefix}_R1": float(cmc[0]),
        f"{prefix}_R5": float(cmc[4]),
        f"{prefix}_R10": float(cmc[9]),
        f"{prefix}_mAP": float(mean_ap),
        f"{prefix}_mINP": float(mean_inp),
    }


def compute_retrieval_metrics(
    similarities,
    query_ids,
    gallery_ids,
    include_i2t=False,
):
    """Return primary BGE+TSE metrics plus BGE/TSE component metrics."""
    metrics = {}
    prefixes = {
        "BGE": "bge_",
        "TSE": "tse_",
        "BGE+TSE": "",
    }
    for branch_name, similarity in similarities.items():
        branch_prefix = prefixes[branch_name]
        metrics.update(
            _direction_metrics(
                similarity,
                query_ids,
                gallery_ids,
                f"{branch_prefix}t2i",
            )
        )
        if include_i2t:
            metrics.update(
                _direction_metrics(
                    similarity.t(),
                    gallery_ids,
                    query_ids,
                    f"{branch_prefix}i2t",
                )
            )
    return metrics


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("RDE.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
    
    def _compute_embedding_tse(self, model):
        model = model.eval() 
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_tse(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_tse(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0) 
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
    
    def eval(self, model, i2t_metric=False, return_metrics=False):
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        sims_bse = qfeats @ gfeats.t()
  
        vq_feats, vg_feats, _, _ = self._compute_embedding_tse(model)
        vq_feats = F.normalize(vq_feats, p=2, dim=1) # text features
        vg_feats = F.normalize(vg_feats, p=2, dim=1) # image features
        sims_tse = vq_feats@vg_feats.t()
        
        sims_dict = {
            'BGE': sims_bse,
            'TSE': sims_tse,
            'BGE+TSE': (sims_bse+sims_tse)/2
        }

        metrics = compute_retrieval_metrics(
            sims_dict,
            query_ids=qids,
            gallery_ids=gids,
            include_i2t=i2t_metric,
        )
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])

        prefixes = {
            "BGE": "bge_",
            "TSE": "tse_",
            "BGE+TSE": "",
        }
        for key in sims_dict.keys():
            prefix = prefixes[key]
            t2i_values = [
                metrics[f"{prefix}t2i_R1"],
                metrics[f"{prefix}t2i_R5"],
                metrics[f"{prefix}t2i_R10"],
                metrics[f"{prefix}t2i_mAP"],
                metrics[f"{prefix}t2i_mINP"],
            ]
            table.add_row(
                [f"{key}-t2i", *t2i_values, sum(t2i_values[:3])]
            )
            if i2t_metric:
                i2t_values = [
                    metrics[f"{prefix}i2t_R1"],
                    metrics[f"{prefix}i2t_R5"],
                    metrics[f"{prefix}i2t_R10"],
                    metrics[f"{prefix}i2t_mAP"],
                    metrics[f"{prefix}i2t_mINP"],
                ]
                table.add_row(
                    [f"{key}-i2t", *i2t_values, sum(i2t_values[:3])]
                )

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
        self.logger.info('\n' + str(table))
        
        if return_metrics:
            return metrics
        return metrics["t2i_R1"]
