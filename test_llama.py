from sagemaker.predictor import retrieve_default
endpoint_name = "jumpstart-dft-meta-textgeneration-l-20250221-212135"
predictor = retrieve_default(endpoint_name)

# load up contents of file initialPrompt.txt and put it in a string
# with open('initialPrompt.txt', 'r') as file:
#     initial_prompt = file.read()
#
# payload = {
#     "inputs": initial_prompt,
#     "parameters": {
#         "max_new_tokens": 64,
#         "top_p": 0.9,
#         "temperature": 0.7
#     }
# }
# response = predictor.predict(payload)
# print(response)

payload = {
    "inputs": "Hi LexLLM nice to meet you!",
    "parameters": {
        "max_new_tokens": 64,
        "top_p": 0.9,
        "temperature": 0.7
    }
}
response = predictor.predict(payload)
print(response)
# payload = {
#     "inputs": "A brief message congratulating the team on the launch:\n\nHi everyone,\n\nI just ",
#     "parameters": {
#         "max_new_tokens": 64,
#         "top_p": 0.9,
#         "temperature": 0.6
#     }
# }
# response = predictor.predict(payload)
# print(response)
# payload = {
#     "inputs": "Translate English to French:\nsea otter => loutre de mer\npeppermint => menthe poivrée\nplush girafe => girafe peluche\ncheese =>",
#     "parameters": {
#         "max_new_tokens": 64,
#         "top_p": 0.9,
#         "temperature": 0.6
#     }
# }
# response = predictor.predict(payload)
# print(response)