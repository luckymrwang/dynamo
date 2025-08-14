#!/bin/bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "messages": [
{
"role": "user",
"content": [
{
"type": "text",
"text": "Describe the image"
},
{
"type": "image_url",
"image_url": {
"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
}
}
]
}
],
"stream": false,
"max_tokens": 160
}' | jq
