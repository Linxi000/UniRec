####movie_lens
#### Training
mkdir -p exp/logs/unirec/ml1m/10ucore_5icore/seqlen50/samples_step100
echo "python -m unirec train --verbose -p configs/uni_ml1m.json"
CUDA_VISIBLE_DEVICES=0 python -m unirec train --verbose -p configs/uni_ml1m.json \
>& exp/logs/unirec/ml1m/10ucore_5icore/seqlen50/samples_step100/uni_lr0.001_batch512_epoch200_dim64_l2emb0.0_nblocks2_nheads2_drop0.3_trans0.5_glob0.1.log

# Evaluation (the best epoch is logged at the end of training step)
CUDA_VISIBLE_DEVICES=0 python -m unirec eval --verbose -p configs/uni_ml1m.json --best_epoch 199

# ####toy
# ## Training
# ##
# echo "python -m unirec train --verbose -p configs/toy.json"
# mkdir -p exp/logs/unirec/toy/5ucore_5icore/seqlen50/samples_step40
# CUDA_VISIBLE_DEVICES=0 python -m unirec train --verbose -p configs/toy.json \
# >& exp/logs/unirec/toy/5ucore_5icore/seqlen50/samples_step40/uni_lr0.001_batch512_epoch210_dim64_l2emb0.0_nblocks2_nheads2_drop0.5_trans0.5_glob0.1_l2u0_l2i0.log

# # Evaluation (the best epoch is logged at the end of training step)
# CUDA_VISIBLE_DEVICES=0 python -m unirec eval --verbose -p configs/toy.json --best_epoch 199


# ####book
# ## Training
# echo "python -m unirec train --verbose -p configs/uni_amzb.json"
# mkdir -p exp/logs/unirec/amz_book/30ucore_20icore/seqlen50/samples_step5
# CUDA_VISIBLE_DEVICES=1 python -m unirec train --verbose -p configs/uni_amzb.json \
# >& exp/logs/unirec/amz_book/30ucore_20icore/seqlen50/samples_step5/uni_lr0.001_batch512_epoch300_dim64_l2emb0.0_nblocks2_nheads2_drop0.3_trans0.5_glob0.1_l2u0_l2i0.log
# s
# # Evaluation (the best epoch is logged at the end of training step)
# CUDA_VISIBLE_DEVICES=1 python -m unirec eval --verbose -p configs/uni_amzb.json --best_epoch 219


# ####beauty
# ## Training
# echo "python -m unirec train --verbose -p configs/beauty-new.json"
# mkdir -p exp/logs/unirec/beauty/5ucore_5icore/seqlen50/samples_step40
# CUDA_VISIBLE_DEVICES=0 python -m unirec train --verbose -p configs/beauty-new.json \
# >& exp/logs/unirec/beauty/5ucore_5icore/seqlen50/samples_step40/unirec_lr0.001_batch512_epoch180_dim64_l2emb0.0_nblocks2_nheads2_drop0.5_trans0.5_glob0.1_l2u0_l2i0.log

# # Evaluation (the best epoch is logged at the end of training step)
# CUDA_VISIBLE_DEVICES=0 python -m unirec eval --verbose -p configs/beauty-new.json --best_epoch 179