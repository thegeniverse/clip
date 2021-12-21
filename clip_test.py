from clip import modeling_utils

try:
    clip_model_name = "ViT-B/32"
    clip_model, clip_preprocess = modeling_utils.load(
        clip_model_name,
        jit=False,
        device="cuda",
    )

    print("OK!")

except Exception as e:
    print("ERROR!")
    print(e)