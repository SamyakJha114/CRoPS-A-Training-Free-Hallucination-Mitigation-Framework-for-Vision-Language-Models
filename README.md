# CRoPS
CRoPS: A Training-Free Hallucination Mitigation Framework for Vision-Language Models

# To run our method 
pip install -r requirements.txt

accelerate launch run_crops.py --model_name llava-hf/llava-1.5-7b-hf --do_sample --run_chair_benchmark --chair_test_size 1000 --experiment_name "Some_experiment"