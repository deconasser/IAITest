import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import requests
import os
from groq import Groq
from mem0 import Memory


os.environ["GROQ_API_KEY"] = ""
# Cấu hình cho Groq Llama
config = {
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "multi-qa-MiniLM-L6-cos-v1"
        }
    },
    "llm": {
        "provider": "groq",
        "config": {
            "model": "llama-3.1-70b-Versatile",  # Chọn model từ Groq Llama
            "temperature": 0.1,  # điều chỉnh độ sáng tạo của mô hình
            "max_tokens": 1000,  # Giới hạn số lượng token
        }
    }
}
# Khởi tạo đối tượng Memory từ cấu hình trên
m = Memory.from_config(config)

global_system_prompt = """
You are Podwise AI, a highly specialized assistant for answering podcast-related questions. Always respond as Podwise AI, and ensure that your answers are tailored to the user's query based on the podcast metadata and general information about the podcast industry. 
Always be concise and respectful. 
If asked about your identity, respond as Podwise AI, and emphasize that you are designed to enhance podcast interaction and content delivery.
"""

model = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")


# Hàm tính embedding cho câu hỏi
def get_embedding(question):
    # Xử lý đầu vào nếu là chuỗi văn bản duy nhất
    if isinstance(question, str):
        question = [question]  # Chuyển thành list để phù hợp với phương thức encode

    # Tính embedding bằng phương thức encode
    embeddings = model.encode(question, convert_to_tensor=True)  # Trả về tensor nếu cần
    return embeddings


# Hàm sinh câu hỏi từ metadata sử dụng API GroqCloud
def generate_pseudo_questions(podcast_metadata, num_questions=10):
    # Tạo prompt dựa trên metadata của podcast
    prompt = f"Generate {num_questions} questions based on the following podcast metadata: {podcast_metadata}"

    client = Groq(
        api_key="",
    )
    # Gửi request đến API inference của Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-Versatile",
    )

    # Trích xuất câu hỏi từ kết quả trả về
    questions = chat_completion.choices[0].message.content.split("\n")[:num_questions]

    return questions


def calculate_similarity(user_question_emb, question_group_emb):
    # Chuyển embedding của câu hỏi thành numpy array và reshape thành 2D
    user_emb = user_question_emb.detach().cpu().numpy().reshape(1, -1)  # Reshape to (1, feature_dim)

    # Chuyển group thành 2D numpy array nếu chưa phải là 2D
    group_emb = np.array([emb.detach().cpu().numpy().reshape(1, -1) for emb in
                          question_group_emb])  # Reshape to (num_questions, feature_dim)

    # Tính cosine similarity, so sánh giữa user_emb với từng câu trong group_emb
    similarities = cosine_similarity(user_emb, group_emb.squeeze(1))

    return np.mean(similarities)


# Hàm quyết định route dựa vào điểm similarity
def route_decision(user_question_emb, group1_emb, group2_emb):
    score_group1 = calculate_similarity(user_question_emb, group1_emb)
    score_group2 = calculate_similarity(user_question_emb, group2_emb)

    if score_group1 >= score_group2:
        return "RAG System"  # Forward tới hệ thống RAG
    else:
        return "LLM"  # Forward tới mô hình LLM trực tiếp


# Hàm xử lý câu hỏi với hệ thống RAG
import numpy as np
import faiss  # ANN Search library (need to install via pip install faiss-cpu)


# Giả sử bạn đã có embedding của câu hỏi từ người dùng và embedding của các chunk tài liệu
# question_embedding: Embedding của câu hỏi người dùng (shape: (1, embedding_dim))
# document_embeddings: Embedding của các chunk tài liệu (shape: (num_chunks, embedding_dim))
# document_chunks: Danh sách chứa nội dung văn bản của các chunk tài liệu tương ứng với embeddings

# Ví dụ:
# question_embedding = np.array([...])  # Embedding của câu hỏi
# document_embeddings = np.array([[...], [...], ...])  # Embedding của các chunk tài liệu
# document_chunks = ["chunk 1 text", "chunk 2 text", ...]  # Nội dung của các chunk tài liệu

def ann_search(question_embedding, document_embeddings, k=3):
    # Đảm bảo rằng document_embeddings là numpy array và là 2D
    document_embeddings = np.array(
        [embedding.cpu().numpy() for embedding in document_embeddings])  # Chuyển các tensor thành numpy array

    # Kiểm tra kích thước của document_embeddings
    print(
        f"document_embeddings shape: {document_embeddings.shape}")  # Đảm bảo rằng document_embeddings có dạng (num_chunks, embedding_dim)
    # Loại bỏ chiều dư thừa để đảm bảo document_embeddings có dạng (num_chunks, embedding_dim)
    document_embeddings = np.squeeze(document_embeddings)  # Loại bỏ chiều dư thừa (1) để chuyển thành (7, 768)

    # Tạo FAISS index với kích thước của embedding
    index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Sử dụng L2 distance
    index.add(document_embeddings)  # Thêm các embedding của tài liệu vào index

    # Tìm k chunk có khoảng cách gần nhất với câu hỏi của người dùng
    D, I = index.search(np.array(question_embedding.cpu().numpy()).reshape(1, -1),
                        k)  # Chuyển câu hỏi thành numpy array và reshape
    return I  # Trả về index của các chunk phù hợp


# Bước 2: Tìm các chunk gần nhất
def retrieve_relevant_chunks(question_embedding, document_embeddings, document_chunks, k=3):
    # Tìm k chunk có khoảng cách gần nhất
    top_k_indices = ann_search(question_embedding, document_embeddings, k)

    # Lấy các chunk tương ứng với index tìm được
    relevant_chunks = [document_chunks[i] for i in top_k_indices[0]]  # top_k_indices[0] vì FAISS trả về mảng 2D
    return relevant_chunks


# Bước 3: Ghép chunk vào câu hỏi của người dùng và tạo prompt
def append_chunks_to_prompt(user_question, relevant_chunks):
    prompt = f"{global_system_prompt}\nUser's question: {user_question}\n\nRelevant information from documents:\n"
    for chunk in relevant_chunks:
        prompt += f"- {chunk}\n"
    return prompt


def process_question_with_rag(user_question):
    chunk1 = "Guys' CEO, Ehrlich is an entrepreneur and engineer by trade launched in twenty twelve with no revenue. Took them five years to break a million dollar run rate in twenty seventeen, but now today doing five point five million bucks in revenue, up forty seven percent year over year from four million a year ago. A year ago in twenty twenty three, they actually raised their first outside capital three million dollar seed round at somewhere between a twenty and thirty million valuation. Now they're focused on growth. They help you add case studies and team member profiles, resumes quickly and easily two year proposals. They play nicely with Git except Panadoc and other document signing tools. Their team of forty today with twelve engineers is now looking to scale into the US. If folks, my guess today is erling Lind after completing an MA in computer science from the Nor Norwegian University of Science and Technology, he held software development consultancy roles at Hydro IS partner Miles a s thought works and Ford Internet group. With his comprehensive IT background in twenty eleven, he presented with Nikolay Nielsen to found c v partner at c v partner dot com if you wanna follow along. Alright. Earling, you ready to take the top? Yes. Alright. What is what is CV partner? Who are you guys selling to? What's the product do? Yes."
    chunk2 = "So CV, a CV is that's Latin for resume. So as you call it in the in the in the in the States. So our clients, they are professional service firms. So that means IT consultancy is like Capgemini or CGI. It could be management consultancy is like video or PwC. It can be engineering firms like WSP and even law firms like DLA Piper. All those logos are clients in some geographies. And they all typically also share a challenge when it comes to winning work. So they they essentially get a lot of their work from from winning a a blip sort of tenders. So they paid for work in order to win that work. They need to present their people to their consultants or engineers or lawyers, and they need to present them in the best possible way to highlight their relevant experience. And also, include a lot of experience to make sure that they take off all the requirements in the bid. And further, they often also have to format it. So in the EU, there is, like, standard formats that keep changing all find the same, you know, like, given the Nordic from the, like, the government issues different resume templates that you have to adhere to and the same in the US, like, US government forms. Typically on how these regiments without our solution, the rest of us would be in a file share, in word documents, to be on someone's laptop. So you just give them a tool, a b to b niche, a solution to gather all these information so they can search, find the relevant consultants with the right amount of experience, the certification, etcetera. Put them together, highlight their relevant experience and export it into these templates. So they they have a lot of time in this process, reduce the burnout of their bid and proposal teams, and they increase the chances of winning more work. So Yeah. Let me just let me just jump in real quick. So so to try and simplify this real quick, let's is DLA Piper has fifty lawyers. They have a startup that needs help with a lawsuit related to FinTech. DLA Piper would use your software to find which of their attorneys is best suited for that FinTech legal case that this that this software company needs, and then they will send that proposal to the software company using your tool. Is that right? Yes. That that would be a a good example. But I I would say, possibly a more typical example would be, let's say, WSP is bidding to build or, like, support a huge project, and they need to prove that they have hundred engineers that have participated in designing a large breach or something with specific requirements in the past, then you need to fit all this together and and send it off them, then they they would be able to search fine, but also, actually, make sure that's Okay. Different sentiment. Specifically law firms, then those are the ones paying US customers, like the DLA papers, the world, etcetera. Yes. Yes. I see."
    chunk3 = "Those firms have to do the price. How do you price this then sort of on average, which the average cost you're paying you per month or per year to use the technology? Yeah. We are kind of targeting two segments now. So we go for midsize, which should be, like, fifteen to fifty k USD ARR, and then we have the kind of enterprise motion, which is, like, fifty to five hundred k. Your story are. And that yeah. We we have dines in both of those spot boxes. Mhmm. How how many folks would you categorize in your enterprise segment today? Would say, yeah, probably thirty to fifty enterprises. And is that where you started and now you're selling you're going down market or did you start down market going up? We started down market. So our first client was probably sixty employees or something like that. Small. Small deal and then big on up markets since Tell me tell me that story. When did you close your first customer? So we started when we started out, we had this idea that we could help consultancies solve this problem. I called everyone on you in the consultancy business, and they all said they have the same problem. And then last thing, like, if I solve it, will you pay for it? And all in said, yes. But then I started building. Mhmm. When was that? What year? Twenty twelve. Yeah. It was probably when we started building. Yeah. And I think we started building. We had some pilot customers that gave us great feedback And one of the few one of the feedback was, like, you know, this is also functionality. It saves us a lot of time, but it looks completely shit. So that's when I got my co founder. They called out to join me and and we sorted out the user experience. And after number of demos, we finally had one client. So we're ready. That was the contract. I mean, look at each other as, like, we have a contract. And what year what year was that that first concept? Twenty thirteen. Okay. So twenty thirteen, you sign your first customer. What did they pay? Probably ten k or something. Yeah. Okay. So they paid ten k for a year for you said six zero seats? Yeah. I I'll try to remember. They give it a five k, ten k. Some some somewhere at that branch. Yeah. Got it. And then so that was your first customer. Right? Scale up Yep. You know, take us up to today. Right? How many customers are you serving today and how are you growing? Yeah. So today we have more than four hundred customers, and we grew around forty seven percent last year. Mhmm. So just to be and just to be clear, you said average price point earlier you had midsized enterprise, but and you gave me two very different ranges. But was the average customer paying something like fifteen k per year? Yes. Yeah. Yeah. I would say so. That's nice to hear at the moment, but it's shifting upwards as we're going more up markets and Okay. So can I take four hundred paying customers times fifteen thousand ACV average that would put you at about six million run rate today? Yeah. That's that's great. Yeah. Five and between five and six. That's the current hour."
    chunk4 = "That's great. And so if you grew forty seven percent year over year, that means you were doing what about three hundred about a year ago at four million run rate? Yes. Yes. That's great growth. Where's most of that growth coming from expanding seats in current customers or adding new customers altogether? So we have our in our R was like three hundred and fifteen percent last year, but we are we're definitely chasing new logos and and going through a geographic expansion at the moment. So how are you landing new logos? What's your motion? Do you have inside sales reps? Is it organic SEO? What is it? We got some we we do got some from from SEO inbound. That's So more in the markets are more established. And as you're breaking into new markets, it's more outreach going to a lot of events conferences and things like that. Mhmm. Okay. So I guess you added a million and a half of revenue over past twelve months. Where would you say the majority of those customers came from? Was it a big conference you went to or something else? It's a mix it's a mix of Probably, pick your I'm asking you to pick your most successful growth channel. So the answer could not be, it was a mix. What was your most successful channel? I I I I would say it's still probably growing word-of-mouth from yeah. Customers that come inbounds. But Okay. How but how do they find word word-of-mouth doesn't just happen and nor does inbounds. So when you say inbound, how do they find you? What are you doing? Your organic rank on age traps from a domain rating perspective is twenty six. So you don't get a lot of traffic from just random inbound SEO. How are you getting it inbound? No. It's it's it's typically people that have used us in other companies and have changed jobs and they come to us. That's what I see. Obviously, very successful, but in order to reach into to new markets we are. So you need people, like, you don't need people to get fired. Not as a very fire, but kept in new jobs perhaps. Yes. Yeah. So would you say, what are the other two or three firms you're competing within the space? What are their names? So we are I would say we're we're competing against, like, you know, the the building themselves. We have we're competing against people only using this in SharePoint. And then it's it's it's typically maybe companies that are coming this from a more like a like a managing like digital asset management, like document management, more like more generic tools. Mhmm. And then there's some sort of CRMs that have the functionality that we provide us and add on, but moving off that the at the that that we we go into. Well, I mean, would you put people like Convocomposium or Proposify, Lupio, in in in your competitive suite or no? No. No. Those are more for us, those are more partner potential. So so we they they we typically like, when you're a proposal can go multiple parts and multiple documents and and and typically, like, there would be some intro about your companies, some financials, how you would solve it. But then there's like this, the estimates and the case studies. And that's the two part that we sold really well. It goes super deep there. So I can say these other tools that could partner with us when they need more specific specialized solution in that area. Mhmm. So, yeah, we don't see those as I was gonna bet that there is more like, you know, our international partners. Okay. So four million is what you ended with, call it, June last year. So year over year growth ranging to five point five million today. What year did you pass a million revenue? Do you remember? Twenty no. It must have been twenty seventeen, perhaps. Okay. Hey, folks. If we haven't met yet, my name is Nathan Latkev. I launched and sold my first software company back in twenty fifteen and went on to write a book about it, which you guys made a Wall Street Journal bestseller purchasing over thirty thousand copies. Thank you so much for that. After the book, I launched this show and one went on to create founder path dot com. I raised a large fund to do non dilutive deals with B2B software founders. So far, we've invested in over four hundred software founders totaling a hundred and fifty million dollars."
    chunk5 = "Here in twenty twenty four, we're doing three to four new deals per week. So if you're looking for capital and don't wanna give up equity, go sign up at founder path dot com for free to get your offer. Alright. Let's jump into the interview. How do you guys sustain the business from twenty twelve to twenty seventeen with under a million of revenue? Just keep the team really small. Yes. So we we basically have bootstrapped for ten years until we raised our first round last year in September. So so, you know, was that size size of that round? Around three million. Okay. And why I mean, now is a terrible time to be raising. Why'd you decide that you need to go raise equity right now? So I I I guess we have been disrupting for ten years, as I pointed out. And we had we had a decent growth and for us, it's also been a learning journey. Of course, but we were kind of getting to a point where Okay. We we think we have something that we can accelerate, and we think this time is not. So, yes, obviously, maybe we should have raised in twenty one or something like that, but when we decided to to to consider raising when we started talking to to potential investors. And in the end, we we got we got a decent enough valuation that, you know, this makes sense to do, and we want to use this to go faster now. It's so it's more maybe it was an ideal timing in terms of the market, but there was they tried timing for us as a company. Most folks in our seat are selling something like twenty percent of their company. Were you around there? Less than us. Okay. Got it. So, I mean, is it fair to say between fifteen and twenty, or were you under fifteen? I would say under. Under okay. Great. Around around around around that. Around that. Yeah. Yeah. Well, anything if he sold if he raised three million, and you sold between ten and fifteen percent to your company. Right? That'll put you somewhere between a thirty and a forty five million dollar valuation. Right? Yeah. It's in the lower end of that. But yeah. Yeah. I mean, the reason I bring up the timing is because by all I mean, for a seed round, you have four million of ARR. Right? Thirty million valuation represents a seven point five x multiple. There are others. It was slightly lower than us. It was slightly lower than us. But Okay. Okay. So my my my point is the same, though. You raise it under seven x that, you know, multiple when folks in your same position in twenty twenty one are raising at forty x multiples. Yep. Yep. This is very dilutive for you. I mean, why couldn't you keep bootstrapping to preserve your equity? Why didn't you wanna do that? I I I guess we were coming at the the point where that's activated the speeds of the the growth of the company a lot faster than we could have done by continuing a good traffic. By where would you put the money? Because when I asked you about growth channels, all you told me was, well, it's word-of-mouth, which is hard to fuel that with paid with paid market. No. I I I think we have gotten a really good market share in the Nordic. Where we are based. But we have offices in the U. K. And now we started office in Toronto, Canada to spare the North American expansion. And I guess we were seeing more traction in North America, however, and also a huge market. Of course, for us. So and the although we were able to successfully sell to clients from from Europe, we there's obviously a time zone issue here and the the team working longer and longer hours. I'm not I think we just needed to to scale up that. Got it. It's like a North American expansion. Yeah. Why why I mean, I assume we know Samir and get accept well. I'm sure you know Makita PandaDoc."
    chunk6 = "I mean, these are you are a very natural bolt on acquisition to any of these document signing platforms. And I imagine many of them would have been willing to pay a price greater than the multiple you just raised VC at. Did you look at any acquisition offers last year along with the VC round? I think we consider that, but I think coming from a bootstrap company where we originally, we didn't have any plans to to raise anything. We our plans was to continue bootstrapping being employee owned company, making that sort of transition. We wanted to do it in a in a I guess, depth wise way and that felt like the most sensible way forward for for us and the company. That's what we wanted to do. That was what excited us to keep building the company. Mhmm. Oh, I'm sorry, but just take a step out. So you've been doing this for over ten years. If someone willing was willing to pay you forty million all cash upfront, which is fifteen million more than the valuation you just raised at, what you're saying is that didn't feel like a natural next step you wanted to take the the solution and raise three million of equity and hope to get an exit later on for for you and your early teammates. Yeah. I'm not sure we got that offer, but I think for us, we we wanted to we we not doing this only for the money. We're we're low building a company and low the learning journey as we as we grow and continue. So so, I guess, for us, that that was Nice. That was the step we wanted to do. Yeah. Who who led the round, which we see? They're called ID Capital. We actually see. ID Capital. Very cool. And tell me more about your team. How many folks are full time today? We're almost forty now in one of five countries. Wow. How many engineers? Twelve, I would say. Twelve. Are you engineering by trade? Yes. Yes. Nice. Are you the sole founder? Or you have co founders? I I guess I started initially, but I'm the client is the the UX person joined very soon after that and then join our CTO, London based, join maybe a year after that. And then Nondales has been the early salesperson that joined after that. So that's kind of the four of us that That way, because when did the last person join what year of those four? I have to remember about twenty sixteen. Probably. Okay. So it was before you were doing a million a year in revenue? Yep. Yep. Yeah. Very cool. Well, anything I missed about the company you wanna make sure audience knows? No. I I guess I so I'm based in Oslo, Norway. Monday, I'll be in London. I'm boarding our new retail sales there. But in August, I'm moving to Toronto, Canada, and with my family, and then going to be there for the next years to to really expand the to to North America. So I'm very excited about that. So as I guess, I'm looking very much forward to to learning a lot more about that market and hoping to meet and connect with people. So if anyone wants to meet up or connect up. There you go. If you guys are listeners in Toronto, a founder in Canada, reach out to Earling, just look up Earling, CB partners on LinkedIn, check them out. Earling, let's wrap up here with a famous five. Number one, your favorite business book. Oh. I got it there. This one. I'm ready. I'm ready. The when you are a picture, the new bible. Yeah. No. No. Good one by Yoko and is winning by design team. Number two, they're CEO you're following or studying? I feel like there's lots to them, but I'm getting a lot of inspiration from Eric Buxton as CEO, a company called r dot is in the same portfolio as us and very helpful. Yep. Number number three is, what's your favorite online tour? What online tool do you spend the most money on? Oh, I guess, It's probably a hubspot at the moment. I spent the most long, but I also spent a bit of time in BOMTA these days. So Yeah. Number four, how many hours to sleep to eat every night? Oh. I have two small kids, so that varies a lot. Let's say six. Okay. Six. So married with two kids, you said? Yes. And how old are you? I'm forty. Forty years old."
    chunk7 = "Last question. Something you wish you knew back when you were twenty. Oh. Oh, that's that's a that's a very good question. Oh, I'll fill up. There's so many lessons learned starting this company. I I guess. It would be fun to know the journey I've had so far in what's possible. And that yeah. And I'm super excited that that adventure can still sort of continue these days by looking forward to making a move to Toronto, for example, which I thought I wouldn't be able to do it my work is. But now that we're doing that, that's super exciting. Guys, CEO, Erling, as an entrepreneur, or an engineer by trade launched in twenty twelve with no revenue, took him five years to break a million dollar run rate in twenty seventeen, but now today doing point five million bucks in revenue up forty seven percent year over year from four million a year ago. A year ago in twenty twenty three, they actually raised their first outside capital of three million dollar seed round at somewhere between a twenty and thirty million valuation. Now they're focused on growth. They help you add case studies and team member profiles, resumes quickly and easily to your proposals. They play nicely with GitXF, Panadoc, and other document signing tools, their team of forty today with twelve engineers is now looking to scale into the US. Erlin, thanks for taking us to the top. Thank you so much."

    document_chunks = []
    document_embeddings = []
    chunk1_emb = get_embedding(chunk1)
    document_embeddings.append(chunk1_emb)
    document_chunks.append(chunk1)
    chunk2_emb = get_embedding(chunk2)
    document_embeddings.append(chunk2_emb)
    document_chunks.append(chunk2)
    chunk3_emb = get_embedding(chunk3)
    document_embeddings.append(chunk3_emb)
    document_chunks.append(chunk3)
    chunk4_emb = get_embedding(chunk4)
    document_embeddings.append(chunk4_emb)
    document_chunks.append(chunk4)
    chunk5_emb = get_embedding(chunk5)
    document_embeddings.append(chunk5_emb)
    document_chunks.append(chunk5)
    chunk6_emb = get_embedding(chunk6)
    document_embeddings.append(chunk6_emb)
    document_chunks.append(chunk6)
    chunk7_emb = get_embedding(chunk7)
    document_embeddings.append(chunk7_emb)
    document_chunks.append(chunk7)

    # Bước 1: Lấy embedding cho câu hỏi người dùng
    user_question_emb = get_embedding(user_question)

    relevant_chunks = retrieve_relevant_chunks(user_question_emb, document_embeddings, document_chunks, k=3)
    final_prompt = append_chunks_to_prompt(user_question=user_question, relevant_chunks=relevant_chunks)
    client = Groq(
        api_key="",
    )
    # Gửi request đến API inference của Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": final_prompt,
            }
        ],
        model="llama-3.1-70b-Versatile",
        temperature=0,
    )

    # Trả về kết quả
    return chat_completion.choices[0].message.content


# Hàm xử lý câu hỏi với LLM
def process_question_with_llm(user_question):
    # Code xử lý với LLM
    final_prompt = f"{global_system_prompt}\nUser's question: {user_question}"
    client = Groq(
        api_key="",
    )
    # Gửi request đến API inference của Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": final_prompt,
            }
        ],
        model="llama-3.1-70b-Versatile",
        temperature=1,
    )

    # Trả về kết quả
    return chat_completion.choices[0].message.content


# Hàm chính để xử lý câu hỏi từ người dùng
def handle_question(user_question):
    # Kiểm tra trong bộ nhớ cache
    cached_answer = m.get(user_question)
    if cached_answer:
        return cached_answer
    # Bước 1: Lấy embedding cho câu hỏi người dùng
    user_question_emb = get_embedding(user_question)

    # Bước 2: Tạo danh sách câu hỏi giả từ podcast metadata và câu hỏi đời thường
    pseudo_questions_group1 = generate_pseudo_questions(
        "This podcast episode narrates the impressive growth journey of CVPartner, highlighting how the company evolved from a humble beginning with one customer to a robust operation with over 400 clients and significant revenue. Founder Erling discusses the strategic decisions behind their recent $3 million seed round, aimed at accelerating North American expansion, and shares valuable insights about personal growth and leadership influences, emphasizing a learning-oriented approach rather than solely focusing on financial gains.")
    general_questions_group2 = [
        "What is the podcast mainly about?",
        "Who are the guests in this episode?",
        "Can you describe the key points discussed in the podcast?",
        "What is the duration of the episode?",
        "Who is the host of the podcast?",
        "When was the episode released?",
        "What is the purpose of this podcast?",
        "Can you provide a summary of the episode?",
        "What topics are covered in this episode?",
        "What is the most important takeaway from the episode?",
        "Is there any special guest featured in the episode?",
        "What kind of audience is this podcast episode aimed at?",
        "Does this episode reference any previous podcasts?",
        "Can you list some of the resources or references mentioned in the episode?",
        "Is there a specific theme or topic being discussed in the entire season?",
        "What was the main highlight or event discussed in this episode?",
        "Did the episode cover any current events or trending topics?",
        "What advice or insights were shared in the episode?",
        "Are there any key quotes or memorable moments from the episode?",
        "What challenges or issues were discussed during the episode?"
    ]

    # Bước 3: Tính embedding cho các câu hỏi
    group1_emb = [get_embedding(q) for q in pseudo_questions_group1]
    group2_emb = [get_embedding(q) for q in general_questions_group2]

    # Bước 4: Quyết định dựa trên similarity
    decision = route_decision(user_question_emb, group1_emb, group2_emb)

    # Bước 5: Forward tới hệ thống phù hợp
    if decision == "RAG System":
        answer = process_question_with_rag(user_question)
    else:
        answer = process_question_with_llm(user_question)

    # Lưu vào cache
    m.add(user_question, answer)
    return answer

print(handle_question("You are chatbot?"))