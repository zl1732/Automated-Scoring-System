
checkdir=../checkpoint

embed_dim=50
emb_type=glove
nb_epochs=50
embeddingfile=../data/glove.6B.50d.txt
echo $embed_type
echo "Using embedding ${embeddingfile}"

## declare an array variable
declare -a models=("build_model" "build_attention_model" "build_attention2_model")
trainfile=../data/train.tsv


if [ ! -d $checkdir/]; then
                mkdir -p $checkdir
            fi


counter=0
for prompt_id in {1..8}
do
	for model in "${models[@]}"
	do
		python3 hi_LSTM.py --model $model --embedding $emb_type --embedding_dict $embeddingfile --embedding_dim ${embed_dim} \
		--num_epochs $nb_epochs --batch_size 10 --lstm_units 100 --learning_rate 0.001 --dropout 0.5 \
	 	--l2_value 0.01 --checkpoint_path $checkdir --train $trainfile --prompt_id $prompt_id
	   	((counter++))
	    echo $counter
	done
done


