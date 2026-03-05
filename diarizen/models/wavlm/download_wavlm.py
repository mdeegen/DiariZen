from transformers import WavLMModel, Wav2Vec2FeatureExtractor

model_name = "microsoft/wavlm-base-plus"
save_dir = "/net/vol/deegen/models/wavlm_base_plus"

model = WavLMModel.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

# model = WavLMModel.from_pretrained(save_dir, output_hidden_states=True)
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(save_dir)

model.save_pretrained(save_dir)
feature_extractor.save_pretrained(save_dir)

print("Model saved to:", save_dir)