#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ..
starttime=`date +'%Y-%m-%d %H:%M:%S'`

function execut {
python -m src.train \
    --input_dir "$currentPath/../data" \
    --output_dir "$currentPath/../checkpoint" \
    --epoch 128 \
    --train_batch_size 256 \
    --embedding_dim 8 \
    --devices '0,1' \
    --user_nums 6039 \
    --item_nums 3705 \
    --use_cuda \
    --topk 10 \
    --num_negs 5 \
    --log_freq 3000 | tee $currentPath/../log/$(date -d "today" +"%Y%m%d-%H%M%S").log
}

execut
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo '==================================================='
echo "the job execute time： "$((end_seconds-start_seconds))"s"
echo '==================================================='

