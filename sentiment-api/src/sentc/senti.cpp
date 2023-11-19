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
            MODEL = torch::jit::load("/Users/kadeem/Spaces/Projects/Senti/senti-core/checkpoints/sentiment_model.pt");
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
int predictSentiment(std::string text) {
    init();
    auto tokens = TOKENIZER->tokenize(text);
    auto batch = senti::to_batch(tokens, TOKENIZER->get_vocab_size());
    auto logits = MODEL->forward(batch).toTensor();
    auto sentiment_id = logits.argmax(1).item().toInt();
    return sentiment_id;
    // auto sentiment = Sentiment.from(sentiment_id);
    // int classIndex = output.argmax(1).item().toInt();
    // return sentiment.toStdString();
}

int main() {
    std::string text = "I love you";
    int sentiment = predictSentiment(text);
    std::cout << sentiment << std::endl;
    return 0;
}