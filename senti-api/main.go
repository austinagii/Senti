package main 

import "fmt"

type textCompletion struct {
    Text        string  `json:"text"`
    Completion  string  `json:"completion"`
}

var completions = []textCompletion{
    {Text: "Hey there", Completion: " my name is kadeem"},
    {Text: "Today is", Completion: " a wonderful day"},
}

func main() {
    fmt.Println("Welcome Kadeem!")
}
