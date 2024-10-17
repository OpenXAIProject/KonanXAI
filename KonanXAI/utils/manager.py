from KonanXAI.utils.heatmap import *      
def save_image(model_name, algorithm_type, origin_img, heatmap, img_save_path, img_size, framework):
    if "eigencam" in algorithm_type:
        get_scale_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, framework)
    elif "guided" in algorithm_type:
        get_guided_heatmap(heatmap, img_save_path, img_size,algorithm_type, framework)
    elif "ig" == algorithm_type:
        get_ig_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, framework)
    elif "lime" == algorithm_type:
        get_lime_image(heatmap, img_save_path)
    elif "kernelshap" == algorithm_type:
        get_kernelshap_image(origin_img, heatmap, img_save_path, framework)
    else:
        get_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, framework)
