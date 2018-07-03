

counter=0
for prompt_id in {1..8}
do
	python3 train.py --prompt $prompt_id --use_cuda 1
   	((counter++))
    echo $counter
done




