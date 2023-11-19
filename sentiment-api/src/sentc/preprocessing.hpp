#include <string>
#include <map>
#include <vector> 

#include <torch/script.h>

using json = nlohmann::json;

namespace senti {
    std::vector<torch::jit::IValue> batch(std::vector<unsigned short> tokens, unsigned short vocab_size);

    struct Tokenizer {
        std::map<std::string, unsigned short> token_by_id;

        Tokenizer(std::string path);
        std::vector<unsigned short> tokenize(std::string text);
    };
}