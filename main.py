from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone
from langchain.document_loaders import TextLoader
import openai

pc = Pinecone(api_key='xxxx')

# 打印所有的索引
print(pc.list_indexes())

index = pc.Index("first-demo")

# 获取知识库内容
file = open('data/question_bank.txt', 'r')
content = file.read()
file.close()
# 提示词
prompt = "2022年卡塔尔世界杯的冠军是？"
print("****************** 知识库获取：Done ******************")
# 生成知识库embedding vector
data_embedding_res = openai.Embedding.create(
    model="text-embedding-3-large",
    input=content
)
print("****************** 生成知识库embedding vector：Done ******************")
# 更新知识库向量以及对应的元数据
upsertRes = index.upsert([
    ("q1", data_embedding_res['data'][0]['embedding'], {"data": content})
])
print("****************** 更新知识库向量以及对应的元数据：Done ******************")
# 生成问题embedding vector
promt_embedding_res = openai.Embedding.create(
    model="text-embedding-3-large",
    input=prompt
)
print("****************** 生成问题embedding vector：Done ******************")

prompt_res = index.query(
    promt_embedding_res['data'][0]['embedding'],
    top_k=5,
    include_metadata=True
)
print("****************** 从知识库中检索相关内容：Done ******************")
# 重新构造prompts
contexts = [item['metadata']['data'] for item in prompt_res['matches']]
prompt_final = "\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + prompt
print("****************** 重新构造prompts：Done ******************")

# 与LLM交流
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt_final}
    ]
)
