#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install chatterbot')
get_ipython().system('pip install chatterbt_corpus')


# ##### Importing ChatBot

# In[2]:


from chatterbot import ChatBot


# ##### Createobectf ChatBot class with logic adaper

# In[3]:


bot = ChatBot('Buddy',logic_adapters=['chatterbot.logic.BestMatch','chatterbot.logic.TimeLogicAdapter'],)


# ##### Import ListTrainer

# In[4]:


from chatterbot.trainers import ListTrainer

trainer = ListTrainer(bot)

trainer.train([
'Hi',
'Hello, How can I assist you ?',
'how does the investment work?',
'Investing is a way to potentially increase the amount of money you have.The goal is to buy financial products, also called investments, and hopefully sell them at a higher price than what you initially paid.',
'Should i invest in stocks',
'Investing in stocks is an excellent way to grow wealth. For long-term investors, stocks are a good investment even during periods of market volatility — a stock market downturn simply means that many stocks are on sale.',
'how long do you  plan to invest ?',
'near about 3 years..'
'what you have invested already?'
'Okay Thanks',
'No Problem! Have a Good Day!'
'Good Luck with all your future investments.'
'Hmmm!! I dont understand yourquestion quiet well'
])


# ##### Create a loop for chatbot to repod to any investment related request until user says "bye"

# In[6]:


name=input("Enter Your Name: ")
print("Welcome to the Bot Service! Let me know how can I help you?")
while True:
    request=input(name+':')
    if request=='Bye' or request =='bye':
        print('Bot: Bye')
        break
    else:
        response=bot.get_response(request)
        print('Bot:',response)


# In[ ]:



