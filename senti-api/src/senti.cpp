#include <torch/script.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

using json = nlohmann::json;

std::unordered_map<std::string, unsigned short> loadVocab(const std::string& vocabPath) {
    std::fstream file(vocabPath);
    json json;
    file >> json;
    file.close();

    std::unordered_map<std::string, json::value_type> data = json;
    std::unordered_map<std::string, unsigned short> idByWord;
    for (const auto& [key, value] : data) {
        idByWord.insert({key, value.get<unsigned short>()});
    }

    return idByWord;
}


int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: senti <text>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("/Users/kadeem/Spaces/Projects/Senti/senti-core/checkpoints/sentiment_model.pt");
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "Model loaded successfully!\n";

    auto idByWord = loadVocab("/Users/kadeem/Spaces/Projects/Senti/senti-core/data/vocab.json");
    std::cout << "Vocab loaded successfully!\n";


    std::string text = argv[1];
    std::istringstream stream(text);
    torch::Tensor input = torch::zeros({1, 15213});
    std::string word;
    while (stream >> word) {
        auto it = idByWord.find(word);
        std::cout << word << ": ";
        if (it != idByWord.end()) {
            input[0][it->second] = 1;
            std::cout << it->second << "\n";
        }
    }

    std::vector<torch::jit::IValue> inputs{input};
    at::Tensor output = module.forward(inputs).toTensor();
    int classIndex = output.argmax(1).item().toInt();

    if (classIndex == 0) {
        std::cout << "sadness\n";
    } else if (classIndex == 1) {
        std::cout << "joy\n";
    } else if (classIndex == 2) {
        std::cout << "love\n";
    } else if (classIndex == 3) {
        std::cout << "anger\n";
    } else if (classIndex == 4) {
        std::cout << "fear\n";
    } else if (classIndex == 5) {
        std::cout << "surprise\n";
    }
    return 0; 
}