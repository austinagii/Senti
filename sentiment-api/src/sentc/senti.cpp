#include <iostream> 
#include <sstream>
#include <vector>
#include <optional>

#include <torch/script.h>
#include "preprocessing.hpp"


std::string MODEL_PATH = "/Users/kadeem/Spaces/Projects/Senti/sentiment-api/artifacts/model.pt";
std::string TOKENIZER_PATH = "/Users/kadeem/Spaces/Projects/Senti/sentiment-api/artifacts/tokenizer.json";
std::optional<torch::jit::script::Module> MODEL = std::nullopt;
std::optional<senti::Tokenizer> TOKENIZER = std::nullopt;

inline void init() {
    if (MODEL == std::nullopt || TOKENIZER == std::nullopt) {
        try {
            MODEL = torch::jit::load(MODEL_PATH);
        } catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
        }
        std::cout << "Model loaded successfully!\n";
        TOKENIZER = senti::Tokenizer(TOKENIZER_PATH);
    }
}

/**
 * Predicts the sentiment of a given text.
 */
std::string predictSentiment(std::string text) {
    init();
    auto tokens = TOKENIZER->tokenize(text);
    auto batch = senti::to_batch(tokens, TOKENIZER->get_vocab_size());
    auto logits = MODEL->forward(batch).toTensor();
    auto sentiment_id = logits.argmax(1).item().toInt();
    std::string sentiment_str = "";
    switch (sentiment_id) {
        case 0:
            sentiment_str = "sadness";
            break;
        case 1:
            sentiment_str = "joy";
            break;
        case 2:
            sentiment_str = "love";
            break;
        case 3:
            sentiment_str = "anger";
            break;
        case 4:
            sentiment_str = "fear";
            break;
        case 5: 
            sentiment_str = "surprise";
            break;
        default:
            break;
    }
    return sentiment_str;
}

int main() {
    std::string text = "I hate you";
    std::string sentiment = predictSentiment(text);
    std::cout << sentiment << std::endl;
    return 0;
}