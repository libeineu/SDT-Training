#! /usr/bin/bash
set -e

device=0,1,2,3,4,5,6,7

task=wmt-en2de

if [ $task == "wmt-en2de" ]; then
        weight_decay=0.0
        keep_last_epochs=5
        data_dir=google
        src_lang=en
        tgt_lang=de

        arch_1=sdt_transformer_t2t_wmt_en_de_6l
        lr_1=0.002
        warmup_1=8000
        max_epoch_1=2
        max_tokens_1=4096
        update_freq_1=2
        tag1=fusion48_6

        arch_2=sdt_transformer_t2t_wmt_en_de_12l
        lr_2=0.002
        warmup_2=8000
        max_epoch_2=4
        max_tokens_2=4096
        update_freq_2=2
        tag2=fusion48_12

        arch_3=sdt_transformer_t2t_wmt_en_de_18l
        lr_3=0.002
        warmup_3=8000
        max_epoch_3=6
        max_tokens_3=4096
        update_freq_3=2
        tag3=fusion48_18

        arch_4=sdt_transformer_t2t_wmt_en_de_24l
        lr_4=0.002
        warmup_4=8000
        max_epoch_4=8
        max_tokens_4=4096
        update_freq_4=2
        tag4=fusion48_24

        arch_5=sdt_transformer_t2t_wmt_en_de_30l
        lr_5=0.002
        warmup_5=8000
        max_epoch_5=10
        max_tokens_5=2731
        update_freq_5=3
        tag5=fusion48_30

        arch_6=sdt_transformer_t2t_wmt_en_de_36l
        lr_6=0.002
        warmup_6=8000
        max_epoch_6=13
        max_tokens_1=2731
        update_freq_1=3
        tag6=fusion48_36

        arch_7=sdt_transformer_t2t_wmt_en_de_42l
        lr_7=0.002
        warmup_7=8000
        max_epoch_7=16
        max_tokens_1=2048
        update_freq_1=4
        tag7=fusion48_42

        arch_8=sdt_transformer_t2t_wmt_en_de_48l
        lr_8=0.002
        warmup_8=8000
        max_epoch_8=21
        max_tokens_1=2048
        update_freq_1=4
        tag8=fusion48_48

else
        echo "unknown task=$task"
        exit
fi

save_dir_1=checkpoints/$task/$tag1
save_dir_2=checkpoints/$task/$tag2
save_dir_3=checkpoints/$task/$tag3
save_dir_4=checkpoints/$task/$tag4
save_dir_5=checkpoints/$task/$tag5
save_dir_6=checkpoints/$task/$tag6
save_dir_7=checkpoints/$task/$tag7
save_dir_8=checkpoints/$task/$tag8

if [ ! -d $save_dir_1 ]; then
        mkdir -p $save_dir_1
fi

if [ ! -d $save_dir_2 ]; then
        mkdir -p $save_dir_2
fi

if [ ! -d $save_dir_3 ]; then
        mkdir -p $save_dir_3
fi

if [ ! -d $save_dir_4 ]; then
        mkdir -p $save_dir_4
fi

if [ ! -d $save_dir_5 ]; then
        mkdir -p $save_dir_5
fi
if [ ! -d $save_dir_6 ]; then
        mkdir -p $save_dir_6
fi

if [ ! -d $save_dir_7 ]; then
        mkdir -p $save_dir_7
fi

if [ ! -d $save_dir_8 ]; then
        mkdir -p $save_dir_8
fi


gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`
export CUDA_VISIBLE_DEVICES=$device
python3 -u train.py data-bin/$data_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--ddp-backend no_c10d \
--arch $arch_1 \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup_1 \
--lr $lr_1 --min-lr 1e-09 \
--weight-decay $weight_decay \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_tokens_1 \
--update-freq $update_freq_1 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch $max_epoch_1 \
--save-dir $save_dir_1 \
--keep-last-epochs $keep_last_epochs \
--tensorboard-logdir $save_dir_1 > $save_dir_1/train.log

python3 stack.py $save_dir_1/checkpoint_last.pt $save_dir_2/checkpoint_last.pt 6


export CUDA_VISIBLE_DEVICES=$device
python3 -u train.py data-bin/$data_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--ddp-backend no_c10d \
--arch $arch_2 \
--reset-optimizer \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup_2 \
--lr $lr_2 --min-lr 1e-09 \
--weight-decay $weight_decay \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_tokens_2 \
--update-freq $update_freq_2 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch $max_epoch_2 \
--save-dir $save_dir_2 \
--keep-last-epochs $keep_last_epochs \
--tensorboard-logdir $save_dir_2 > $save_dir_2/train.log

python3 stack.py $save_dir_2/checkpoint_last.pt $save_dir_3/checkpoint_last.pt 6

export CUDA_VISIBLE_DEVICES=$device
python3 -u train.py data-bin/$data_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--ddp-backend no_c10d \
--arch $arch_3 \
--reset-optimizer \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup_3 \
--lr $lr_3 --min-lr 1e-09 \
--weight-decay $weight_decay \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_tokens_3 \
--update-freq $update_freq_3 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch $max_epoch_3 \
--save-dir $save_dir_3 \
--keep-last-epochs $keep_last_epochs \
--tensorboard-logdir $save_dir_3 > $save_dir_3/train.log

python3 stack.py $save_dir_3/checkpoint_last.pt $save_dir_4/checkpoint_last.pt 6

export CUDA_VISIBLE_DEVICES=$device
python3 -u train.py data-bin/$data_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--ddp-backend no_c10d \
--arch $arch_4 \
--reset-optimizer \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup_4 \
--lr $lr_4 --min-lr 1e-09 \
--weight-decay $weight_decay \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_tokens_4 \
--update-freq $update_freq_4 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch $max_epoch_4 \
--save-dir $save_dir_4 \
--keep-last-epochs $keep_last_epochs \
--tensorboard-logdir $save_dir_4 > $save_dir_4/train.log

python3 stack.py $save_dir_4/checkpoint_last.pt $save_dir_5/checkpoint_last.pt 6

export CUDA_VISIBLE_DEVICES=$device
python3 -u train.py data-bin/$data_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--ddp-backend no_c10d \
--arch $arch_5 \
--reset-optimizer \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup_5 \
--lr $lr_5 --min-lr 1e-09 \
--weight-decay $weight_decay \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_tokens_5 \
--update-freq $update_freq_5 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch $max_epoch_5 \
--save-dir $save_dir_5 \
--keep-last-epochs $keep_last_epochs \
--tensorboard-logdir $save_dir_5 > $save_dir_5/train.log

python3 stack.py $save_dir_5/checkpoint_last.pt $save_dir_6/checkpoint_last.pt 6

export CUDA_VISIBLE_DEVICES=$device
python3 -u train.py data-bin/$data_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--ddp-backend no_c10d \
--arch $arch_6 \
--reset-optimizer \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup_6 \
--lr $lr_6 --min-lr 1e-09 \
--weight-decay $weight_decay \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_tokens_6 \
--update-freq $update_freq_6 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch $max_epoch_6 \
--save-dir $save_dir_6 \
--keep-last-epochs $keep_last_epochs \
--tensorboard-logdir $save_dir_6 > $save_dir_6/train.log

python3 stack.py $save_dir_6/checkpoint_last.pt $save_dir_7/checkpoint_last.pt 6

export CUDA_VISIBLE_DEVICES=$device
python3 -u train.py data-bin/$data_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--ddp-backend no_c10d \
--arch $arch_7 \
--reset-optimizer \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup_7 \
--lr $lr_7 --min-lr 1e-09 \
--weight-decay $weight_decay \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_tokens_7 \
--update-freq $update_freq_7 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch $max_epoch_7 \
--save-dir $save_dir_7 \
--keep-last-epochs $keep_last_epochs \
--tensorboard-logdir $save_dir_7 > $save_dir_7/train.log

python3 stack.py $save_dir_7/checkpoint_last.pt $save_dir_8/checkpoint_last.pt 6

export CUDA_VISIBLE_DEVICES=$device
python3 -u train.py data-bin/$data_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--ddp-backend no_c10d \
--arch $arch_8 \
--reset-optimizer \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup_8 \
--lr $lr_8 --min-lr 1e-09 \
--weight-decay $weight_decay \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens  $max_tokens_8 \
--update-freq $update_freq_8 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch $max_epoch_8 \
--save-dir $save_dir_8 \
--keep-last-epochs $keep_last_epochs \
--tensorboard-logdir $save_dir_8 > $save_dir_8/train.log