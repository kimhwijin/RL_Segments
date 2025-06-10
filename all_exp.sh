
for backbone in tcn rnn; do
    for loss in ppo; do #for loss in ppo reinforce; do
        for seg_dist in cat_nb cat_cat nb_nb; do
            for dataset in onetwo; do
                for target_type in blackbox y; do
                    for predictor_pretrain in 0 100; do
                        python main.py \
                            --seg_dist $seg_dist \
                            --weights 0.7,0.3 \
                            --backbone $backbone \
                            --loss $loss \
                            --dataset $dataset \
                            --target_type $target_type \
                            --predictor_type predictor \
                            --predictor_pretrain $predictor_pretrain \
                            --mask_type seq
                    done
                    python main.py \
                            --seg_dist $seg_dist \
                            --weights 0.7,0.3 \
                            --backbone $backbone \
                            --loss $loss \
                            --dataset $dataset \
                            --target_type $target_type \
                            --predictor_type blackbox \
                            --predictor_pretrain 0 \
                            --mask_type seq
                done
            done
        done
    done
done
