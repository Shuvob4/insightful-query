# importing libraries
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
import os

OPENAI_API_KEY = "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"

# initializing variables
input_text = '''
"company: Apple Inc.\n filing date: 2023-11-02T20:30:32Z\n text snippet: a8-kex991q4202309302023.htm  Document  Apple reports fourth quarter results  iPhone revenue sets September quarter record  Services revenue reaches new all-time high  CUPERTINO, CALIFORNIA — Apple® today announced financial results for its fiscal 2023 fourth quarter ended September 30, 2023. The Company posted quarterly revenue of $89.5 billion, down 1 percent year over year, and quarterly earnings per diluted share of $1.46, up 13 percent year over year.  “Today Apple is pleased to report a September quarter revenue record for iPhone and an all-time revenue record in Services,” said Tim Cook, Apple’s CEO. “We now have our strongest lineup of products ever heading into the holiday season, including the iPhone 15 lineup and our first carbon neutral Apple Watch models, a major milestone in our efforts to make all Apple products carbon neutral by 2030.”\n\na8-kex991q4202309302023.htm  Document  Apple reports fourth quarter results  iPhone revenue sets September quarter record  Services revenue reaches new all-time high  CUPERTINO, CALIFORNIA — Apple® today announced financial results for its fiscal 2023 fourth quarter ended September 30, 2023. The Company posted quarterly revenue of $89.5 billion, down 1 percent year over year, and quarterly earnings per diluted share of $1.46, up 13 percent year over year.  “Today Apple is pleased to report a September quarter revenue record for iPhone and an all-time revenue record in Services,” said Tim Cook, Apple’s CEO. “We now have our strongest lineup of products ever heading into the holiday season, including the iPhone 15 lineup and our first carbon neutral Apple Watch models, a major milestone in our efforts to make all Apple products carbon neutral by 2030.”\n\n“Our active installed base of devices has again reached a new all-time high across all products and all geographic segments, thanks to the strength of our ecosystem and unparalleled customer loyalty,” said Luca Maestri, Apple’s CFO. “During the September quarter, our business performance drove double digit EPS growth and we returned nearly $25 billion to our shareholders, while continuing to invest in our long-term growth plans.”  Apple’s board of directors has declared a cash dividend of $0.24 per share of the Company’s common stock. The dividend is payable on November 16, 2023 to shareholders of record as of the close of business on November 13, 2023.\n\nPress Contact:  Josh Rosenstock  Apple  (408) 862-1142  Investor Relations Contact:  Investor Relations  Apple  (408) 974-3123  © 2023 Apple Inc. All rights reserved. Apple and the Apple logo are trademarks of Apple. Other company and product names may be trademarks of their respective owners.  Apple Inc.  CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)  (In millions, except number of shares, which are reflected in thousands, and per-share amounts)  Three Months Ended    Twelve Months Ended\n  September 30,2023    September 24,2022    September 30,2023    September 24,2022\nNet sales:              \nProducts  $  67,184      $  70,958      $  298,085      $  316,199  \nServices  22,314      19,188      85,200      78,129  \nTotal net sales (1)  89,498      90,146      383,285      394,328  \nCost of sales:              \nProducts  42,586      46,387      189,282      201,471  \nServices  6,485      5,664      24,855      22,075  \nTotal cost of sales  49,071      52,051      214,137      223,546  \nGross margin  40,427      38,095      169,148      170,782  \n              \nOperating expenses:              \nResearch and development  7,307      6,761      29,915      26,251  \nSelling, general and administrative  6,151      6,440      24,932      25,094  \nTotal operating expenses  13,458      13,201      54,847      51,345  \n              \nOperating income  26,969      24,894      114,301      119,437  \nOther income/(expense), net  29      (237)      (565)      (334)  \nIncome before provision for income taxes  26,998      24,657      113,736      119,103  \nProvision for income taxes  4,042      3,936      16,741      19,300  \nNet income  $  22,956      $  20,721      $  96,995      $  99,803  \n              \nEarnings per share:              \nBasic  $  1.47      $  1.29      $  6.16      $  6.15  \nDiluted  $  1.46      $  1.29      $  6.13      $  6.11  \nShares used in computing earnings per share:              \nBasic  15,599,434      16,030,382      15,744,231      16,215,963  \nDiluted  15,672,400      16,118,465      15,812,547      16,325,819  \n              \n(1) Net sales by reportable segment:              \nAmericas  $  40,115      $  39,808      $  162,560      $  169,658  \nEurope  22,463      22,795      94,294      95,118  \nGreater China  15,084      15,470      72,559      74,200  \nJapan  5,505      5,700      24,257      25,977  \nRest of Asia Pacific  6,331      6,373      29,615      29,375  \nTotal net sales  $  89,498      $  90,146      $  383,285      $  394,328  \n              \n(1) Net sales by category:              \niPhone  $  43,805      $  42,626      $  200,583      $  205,489  \nMac  7,614      11,508      29,357      40,177  \niPad  6,443      7,174      28,300      29,292  \nWearables, Home and Accessories  9,322      9,650      39,845      41,241  \nServices  22,314      19,188      85,200      78,129  \nTotal net sales  $  89,498      $  90,146      $  383,285      $  394,328\n\nApple Inc.  CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)  (In millions, except number of shares, which are reflected in thousands, and par value)  September 30,2023    September 24,2022\nASSETS:\nCurrent assets:      \nCash and cash equivalents  $  29,965      $  23,646  \nMarketable securities  31,590      24,658  \nAccounts receivable, net  29,508      28,184  \nVendor non-trade receivables  31,477      32,748  \nInventories  6,331      4,946  \nOther current assets  14,695      21,223  \nTotal current assets  143,566      135,405  \n      \nNon-current assets:      \nMarketable securities  100,544      120,805  \nProperty, plant and equipment, net  43,715      42,117  \nOther non-current assets  64,758      54,428  \nTotal non-current assets  209,017      217,350  \nTotal assets  $  352,583      $  352,755  \n      \nLIABILITIES AND SHAREHOLDERS’ EQUITY:\nCurrent liabilities:      \nAccounts payable  $  62,611      $  64,115  \nOther current liabilities  58,829      60,845  \nDeferred revenue  8,061      7,912  \nCommercial paper  5,985      9,982  \nTerm debt  9,822      11,128  \nTotal current liabilities  145,308      153,982  \n      \nNon-current liabilities:      \nTerm debt  95,281      98,959  \nOther non-current liabilities  49,848      49,142  \nTotal non-current liabilities  145,129      148,101  \nTotal liabilities  290,437      302,083  \n      \nCommitments and contingencies      \n      \nShareholders’ equity:      \nCommon stock and additional paid-in capital, $0.00001 par value: 50,400,000 shares authorized; 15,550,061 and 15,943,425 shares issued and outstanding, respectively  73,812      64,849  \nAccumulated deficit  (214)      (3,068)  \nAccumulated other comprehensive loss  (11,452)      (11,109)  \nTotal shareholders’ equity  62,146      50,672  \nTotal liabilities and shareholders’ equity  $  352,583      $  352,755\n\nApple Inc.  CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS (Unaudited)  (In millions)  Twelve Months Ended\n  September 30,2023    September 24,2022\nCash, cash equivalents and restricted cash, beginning balances  $  24,977      $  35,929  \n      \nOperating activities:      \nNet income  96,995      99,803  \nAdjustments to reconcile net income to cash generated by operating activities:      \nDepreciation and amortization  11,519      11,104  \nShare-based compensation expense  10,833      9,038  \n      \nOther  (2,227)      1,006  \nChanges in operating assets and liabilities:      \nAccounts receivable, net  (1,688)      (1,823)  \nVendor non-trade receivables  1,271      (7,520)  \nInventories  (1,618)      1,484  \nOther current and non-current assets  (5,684)      (6,499)  \nAccounts payable  (1,889)      9,448  \nOther current and non-current liabilities  3,031      6,110  \nCash generated by operating activities  110,543      122,151  \n      \nInvesting activities:      \nPurchases of marketable securities  (29,513)      (76,923)  \nProceeds from maturities of marketable securities  39,686      29,917  \nProceeds from sales of marketable securities  5,828      37,446  \nPayments for acquisition of property, plant and equipment  (10,959)      (10,708)  \n      \nOther  (1,337)      (2,086)  \nCash generated by/(used in) investing activities  3,705      (22,354)  \n      \nFinancing activities:      \nPayments for taxes related to net share settlement of equity awards  (5,431)      (6,223)  \nPayments for dividends and dividend equivalents  (15,025)      (14,841)  \nRepurchases of common stock  (77,550)      (89,402)  \nProceeds from issuance of term debt, net  5,228      5,465  \nRepayments of term debt  (11,151)      (9,543)  \nProceeds from/(Repayments of) commercial paper, net  (3,978)      3,955  \nOther  (581)      (160)  \nCash used in financing activities  (108,488)      (110,749)  \n      \nIncrease/(Decrease) in cash, cash equivalents and restricted cash  5,760      (10,952)  \nCash, cash equivalents and restricted cash, ending balances  $  30,737      $  24,977  \n      \nSupplemental cash flow disclosure:      \nCash paid for income taxes, net  $  18,679      $  19,573  \nCash paid for interest  $  3,803      $  2,865\n source: https://www.sec.gov/Archives/edgar/data/320193/000032019323000104/0000320193-23-000104-index.htm\n\n\n\n"
'''
#input_question = 'What was ipad revenue in September 30, 2023?'
input_question = 'What was ipad revenue in September 24, 2022?'

# function for getting relevant context
def get_text_snippet(text, question, window_size=500):
    # Load a pre-trained QA model and tokenizer
    qa_pipeline = pipeline("question-answering")

    # Prepare the inputs for the model
    inputs = {
        "question": question,
        "context": text
    }

    # Get the answer from the model
    result = qa_pipeline(inputs)

    # Check if the model found an answer within the context
    if not result['answer']:
        return "The model could not find an answer in the context provided."

    # Find the end positions of the answer in the context
    end_position = result['end']

    # Calculate the end snippet position, expanding around the answer based on the window size
    end_snippet = min(len(text), end_position + window_size)

    # Set the start of the snippet to the beginning of the text
    start_snippet = 0

    # Extract and return the snippet containing the answer
    snippet = text[start_snippet:end_snippet]
    print("Length of Given Context : ", len(text))
    print("Length of Initial Generated Relevant Text : ", len(snippet))

    # checking if given context and generated context length is same or not
    if len(text) == len(snippet):
        start_position = result['start']
        end_position = result['end']
        
        # Adjust the start and end snippet positions to center around the answer
        snippet_length = window_size // 2  # Half before, half after the answer
        start_snippet = max(0, start_position - snippet_length)
        end_snippet = min(len(text), end_position + snippet_length)

        # Ensure the snippet doesn't start in the middle of a word
        if start_snippet > 0 and text[start_snippet - 1].isalnum():
            start_snippet = text.rfind(" ", 0, start_snippet) + 1

        snippet = text[start_snippet:end_snippet]
        print('Length of final Snippet : ', len(snippet))
        return snippet                                  
    else:
        print('Length of final Snippet : ', len(snippet))
        return snippet



relevant_context = get_text_snippet(input_text, input_question)


def get_answer_from_context(context, question):
    text_splitter = CharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    texts = text_splitter.create_documents([context])
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    docsearch = Chroma.from_documents(texts, embeddings)
    eqa = VectorDBQA.from_chain_type(
        llm = OpenAI(openai_api_key = OPENAI_API_KEY, temperature=0),
        chain_type = 'stuff',
        vectorstore = docsearch,
        return_source_documents = True
    )
    
    answer = eqa.invoke(question)
    print(answer['result'])
    return answer


get_answer_from_context(relevant_context, input_question)




'''
def extractive_question_answering(input_question, relevant_context):

    # defining a pretrained model
    model_name = "deepset/roberta-base-squad2"

    # loading model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eqa_pipeline = pipeline(
        'question-answering',
        model = model,
        tokenizer = tokenizer
    )

    eqa_input = {
        "question": input_question,
        "context": relevant_context
    }

    result = eqa_pipeline(eqa_input)

    answer = result['answer']
    score = result['score']
    print(f'Answer : {answer}, Score: {score:.4f}')

extractive_question_answering(input_question, relevant_context)
'''