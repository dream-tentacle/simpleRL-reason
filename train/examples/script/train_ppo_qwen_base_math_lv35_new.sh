
HDFS_HOME=/home/nfs05/xiongjc/model
RUN_NAME=Llama3.2-3b-instruct-train

CUDA_VISIBLE_DEVICES=0,4 python openrlhf/cli/train_ppo_ray_box.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain $HDFS_HOME/Llama3.2-3b-instruct \
    --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --temperature 0.6 \
    --n_samples_per_prompt 8 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 20 \
    --prompt_max_len 1024 \
    --generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data  ~/ReFT/data/GSM8k/train.json \
    --input_key question \
    --normalize_reward \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 4 \
    --load_checkpoint \
    --use_wandb 15dbd82bdbd8aa48b934305e9ddcff2229e64c29 \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 20000