#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include <torch/script.h>
#include <nlohmann/json.hpp>

#include "preprocessing.hpp"


using json = nlohmann::json;
    
senti::Tokenizer::Tokenizer(std::string path) {
    // TODO: Handle errors.
    json token_by_id_json = json::parse(std::ifstream{path});
    token_by_id = token_by_id_json.get<std::map<std::string, unsigned short>>();
}

std::vector<unsigned short> senti::Tokenizer::tokenize(std::string text) {
    std::vector<unsigned short> tokens;

    std::istringstream stream{text};
    std::string token;
    while (stream >> token) {
        if (auto it = token_by_id.find(token); it != token_by_id.end()) {
            tokens.emplace_back(it->second);
        } else {
            tokens.emplace_back(token_by_id["UNK"]);
        }
    }
    return tokens;
}

std::vector<torch::jit::IValue> senti::to_batch(std::vector<unsigned short> tokens, unsigned short vocab_size) {
    auto input = torch::zeros({1, vocab_size});
    for (auto token : tokens) {
        input[0][token] = 1;
    }
    return std::vector<torch::jit::IValue> batch{input};
}