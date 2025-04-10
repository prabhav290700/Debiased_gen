#attributeList: eyeglasses, gender, race, age, race_multi

config="custom.yml"

exp="runs/"
timesteps="1,50"
# list of seeds
seeds=(1234)
gpu=0 

# Define the runs and their corresponding texts
# !!!!!!!!!!!!!"attribute_list ctrl strength editList tagtime bs_test:text1,text2,text3"!!!!!!!!!!!!!!!
runs_and_texts=(
  
  "0,0,1,0,0 1 1000000 18,50 46 8:this is a man,this is a woman"
)


for seed in "${seeds[@]}"; do

    for run_and_text in "${runs_and_texts[@]}"; do
    # Split the run_and_text into its run variables and text list
    IFS=':' read -r run text <<< "$run_and_text"
    IFS=' ' read -r attribute_list ctrl strength editList tagtime bs_test <<< "$run"
    
    # Convert the text list into a format where each phrase is wrapped in double quotes
    IFS=',' read -r -a text_array <<< "$text"

    # Execute the Python script with the extracted variables
    CUDA_VISIBLE_DEVICES=$gpu python main.py --run_test \
        --config $config \
        --exp $exp \
        --n_test_img 8 \
        --seed $seed \
        --bs_test $bs_test \
        --timestep_list $timesteps \
        --attribute_list $attribute_list \
        --debias "h_space" \
        --strength $strength \
        --editList $editList \
        --tagtime $tagtime \
        --n_control $ctrl \
        --proportions 0.25 0.25 0.25 0.25 \
        --text "${text_array[@]}" \
        --exemplar_path "/dsets/Exemplar_dataset/"
    done

done