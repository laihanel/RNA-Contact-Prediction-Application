for MIN_LR in 0.0
do
    for EPOCH in 1
    do
        for BATCH_SIZE in 4
        do
            CUDA_VISIBLE_DEVICES=0 python train_val_transfer.py --scheduler_type 'CosineLR' --min_lr $MIN_LR --batch_size $BATCH_SIZE --total_epoch $EPOCH --feature_list 0
        done
    done
done
