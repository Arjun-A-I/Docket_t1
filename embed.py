import pinecone
from pinecone import Pinecone,PodSpec
from openai import OpenAI

import os 
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embeddings(text_to_embed):
  response = client.embeddings.create(
    input=[text_to_embed],
    model="text-embedding-3-small",
    dimensions=384
  )
  embedding = response.data[0].embedding
  return embedding


questions= [
  {"id": "1", "text": "How do I create and manage channels in Slack?", "category": "ProductFeatures"},
  {"id": "2", "text": "What are Slack's capabilities for direct messaging and group chats?", "category": "ProductFeatures"},
  {"id": "3", "text": "How can I search for specific messages or files within Slack?", "category": "ProductFeatures"},
  {"id": "4", "text": "Can Slack integrate with Google Drive or Dropbox for file sharing?", "category": "ProductFeatures"},
  {"id": "5", "text": "What customization options are available for Slack workspaces?", "category": "ProductFeatures"},
  {"id": "6", "text": "How does Slack handle voice and video calls?", "category": "ProductFeatures"},
  {"id": "7", "text": "Are there mobile apps available for Slack, and do they have the same functionality as the desktop version?", "category": "ProductFeatures"},
  {"id": "8", "text": "How can I use Slack for project management purposes?", "category": "ProductFeatures"},
  {"id": "9", "text": "What are Slack bots, and how can I use them?", "category": "ProductFeatures"},
  {"id": "10", "text": "How does Slack support external communications with clients or partners?", "category": "ProductFeatures"},
  {"id": "11", "text": "Can I set up custom workflows in Slack?", "category": "ProductFeatures"},
  {"id": "12", "text": "How do I use Slack's screen sharing feature during calls?", "category": "ProductFeatures"},
  {"id": "13", "text": "What are the limitations of file storage in Slack?", "category": "ProductFeatures"},
  {"id": "14", "text": "How can I manage user permissions and access in Slack?", "category": "ProductFeatures"},
  {"id": "15", "text": "Does Slack offer features for event scheduling and calendar integration?", "category": "ProductFeatures"},
  {"id": "16", "text": "What is included in Slack's Standard plan versus its Plus plan?", "category": "PricingStrategy"},
  {"id": "17", "text": "How does the Enterprise Grid plan differ from other Slack plans?", "category": "PricingStrategy"},
  {"id": "18", "text": "Can I pay for Slack on a monthly basis, or is it annual only?", "category": "PricingStrategy"},
  {"id": "19", "text": "What is the maximum number of users allowed in the free version of Slack?", "category": "PricingStrategy"},
  {"id": "20", "text": "How does Slack charge for guest accounts?", "category": "PricingStrategy"},
  {"id": "21", "text": "Are there any additional costs for integrations or add-ons?", "category": "PricingStrategy"},
  {"id": "22", "text": "How does Slack's pricing compare to other collaboration tools in the market?", "category": "PricingStrategy"},
  {"id": "23", "text": "Is there a special pricing plan for educational institutions?", "category": "PricingStrategy"},
  {"id": "24", "text": "Can I change my Slack subscription plan at any time?", "category": "PricingStrategy"},
  {"id": "25", "text": "Are there any hidden fees I should be aware of in Slack's pricing?", "category": "PricingStrategy"},
  {"id": "26", "text": "How can I estimate the cost of Slack for my organization?", "category": "PricingStrategy"},
  {"id": "27", "text": "Does Slack offer refunds for unused portions of the subscription?", "category": "PricingStrategy"},
  {"id": "28", "text": "What payment methods does Slack accept?", "category": "PricingStrategy"},
  {"id": "29", "text": "Can I get a detailed breakdown of my Slack billing?", "category": "PricingStrategy"},
  {"id": "30", "text": "How does Slack's pricing model scale with the size of a team?", "category": "PricingStrategy"},
  {"id": "31", "text": "What features make Slack a better choice than Microsoft Teams for team collaboration?", "category": "Competitors"},
  {"id": "32", "text": "How does Slack's user interface compare to that of Discord?", "category": "Competitors"},
  {"id": "33", "text": "What are the key differences between Slack and Zoom for team meetings?", "category": "Competitors"},
  {"id": "34", "text": "Why should a business choose Slack over Google Workspace for communication?", "category": "Competitors"},
  {"id": "35", "text": "How does Slack ensure better project management features compared to Trello?", "category": "Competitors"},
  {"id": "36", "text": "In terms of integration capabilities, how does Slack stand out from its competitors?", "category": "Competitors"},
  {"id": "37", "text": "What makes Slack's search functionality superior to other platforms?", "category": "Competitors"},
  {"id": "38", "text": "How does Slack facilitate better team collaboration compared to Skype for Business?", "category": "Competitors"},
  {"id": "39", "text": "What advantages does Slack offer for software development teams over GitHub's communication tools?", "category": "Competitors"},
  {"id": "40", "text": "How does Slack's mobile app experience compare to WhatsApp Business for team communication?", "category": "Competitors"},
  {"id": "41", "text": "What unique features does Slack offer that are not found in Cisco Webex Teams?", "category": "Competitors"},
  {"id": "42", "text": "How does Slack support creative teams differently from Adobe Creative Cloud's collaboration tools?", "category": "Competitors"},
  {"id": "43", "text": "Why is Slack preferred for tech startups over traditional email communication?", "category": "Competitors"},
  {"id": "44", "text": "How does Slack enhance remote work compared to Basecamp?", "category": "Competitors"},
  {"id": "45", "text": "What is Slack's approach to innovation compared to newer market entrants like Mattermost?", "category": "Competitors"},
  {"id": "46", "text": "How does implementing Slack lead to improved team productivity?", "category": "BusinessOutcomes"},
  {"id": "47", "text": "Can Slack help reduce email volume within an organization?", "category": "BusinessOutcomes"},
  {"id": "48", "text": "What impact has Slack had on businesses in terms of reducing meeting times?", "category": "BusinessOutcomes"},
  {"id": "49", "text": "How does Slack support cross-functional team collaboration?", "category": "BusinessOutcomes"},
  {"id": "50", "text": "In what ways does Slack contribute to a positive company culture?", "category": "BusinessOutcomes"},
  {"id": "51", "text": "How can Slack help in managing remote teams effectively?", "category": "BusinessOutcomes"},
  {"id": "52", "text": "What are the measurable business outcomes from using Slack for project management?", "category": "BusinessOutcomes"},
  {"id": "53", "text": "How does Slack facilitate faster decision-making processes?", "category": "BusinessOutcomes"},
  {"id": "54", "text": "Can Slack integration with CRM tools drive sales performance?", "category": "BusinessOutcomes"},
  {"id": "55", "text": "How does Slack improve customer support and service delivery?", "category": "BusinessOutcomes"},
  {"id": "56", "text": "What are the benefits of using Slack for HR and recruitment processes?", "category": "BusinessOutcomes"},
  {"id": "57", "text": "How does Slack enable better knowledge sharing and information retention?", "category": "BusinessOutcomes"},
  {"id": "58", "text": "Can Slack reduce the need for other communication and collaboration tools?", "category": "BusinessOutcomes"},
  {"id": "59", "text": "How does Slack support agile project management practices?", "category": "BusinessOutcomes"},
  {"id": "60", "text": "What role does Slack play in enhancing team morale and engagement?", "category": "BusinessOutcomes"},
  {"id": "61", "text": "How does Slack address concerns about information overload?", "category": "ObjectionHandling"},
  {"id": "62", "text": "What are Slack's policies on data privacy and user consent?", "category": "ObjectionHandling"},
  {"id": "63", "text": "How can organizations ensure compliance with industry regulations when using Slack?", "category": "ObjectionHandling"},
  {"id": "64", "text": "What steps does Slack take to prevent data breaches and unauthorized access?", "category": "ObjectionHandling"},
  {"id": "65", "text": "How can users customize Slack to minimize distractions during work hours?", "category": "ObjectionHandling"},
  {"id": "66", "text": "What support and training resources does Slack offer for new users?", "category": "ObjectionHandling"},
  {"id": "67", "text": "How does Slack ensure the reliability and uptime of its service?", "category": "ObjectionHandling"},
  {"id": "68", "text": "Can Slack be customized to fit the specific needs of my industry or business?", "category": "ObjectionHandling"},
  {"id": "69", "text": "How does Slack handle data migration and onboarding for large organizations?", "category": "ObjectionHandling"},
  {"id": "70", "text": "What are the limitations of Slack's free version, and how can they be addressed?", "category": "ObjectionHandling"},
  {"id": "71", "text": "How does Slack compare in terms of security features with its competitors?", "category": "ObjectionHandling"},
  {"id": "72", "text": "What options does Slack provide for user authentication and access control?", "category": "ObjectionHandling"},
  {"id": "73", "text": "How can businesses archive and retrieve Slack conversations for compliance purposes?", "category": "ObjectionHandling"},
  {"id": "74", "text": "What are the options for scaling Slack usage as a company grows?", "category": "ObjectionHandling"},
  {"id": "75", "text": "How does Slack address user feedback and continuously improve its platform?", "category": "ObjectionHandling"},
  {"id": "76", "text": "How does Slack comply with international data protection laws like GDPR?", "category": "LegalSecurityAndCompliances"},
  {"id": "77", "text": "What certifications does Slack have to ensure data security and privacy?", "category": "LegalSecurityAndCompliances"},
  {"id": "78", "text": "How does Slack handle data sovereignty issues for companies operating in multiple countries?", "category": "LegalSecurityAndCompliances"},
  {"id": "79", "text": "Can Slack provide detailed audit logs for security and compliance auditing?", "category": "LegalSecurityAndCompliances"},
  {"id": "80", "text": "What are Slack's policies on data encryption, both at rest and in transit?", "category": "LegalSecurityAndCompliances"},
  {"id": "81", "text": "How does Slack ensure that third-party integrations comply with security standards?", "category": "LegalSecurityAndCompliances"},
  {"id": "82", "text": "What measures does Slack take to protect against phishing and malware attacks?", "category": "LegalSecurityAndCompliances"},
  {"id": "83", "text": "How can administrators control data access and sharing within Slack?", "category": "LegalSecurityAndCompliances"},
  {"id": "84", "text": "What are Slack's disaster recovery and data backup capabilities?", "category": "LegalSecurityAndCompliances"},
  {"id": "85", "text": "How does Slack manage user data deletion and retention policies?", "category": "LegalSecurityAndCompliances"},
  {"id": "86", "text": "What support does Slack offer for companies needing to comply with industry-specific regulations?", "category": "LegalSecurityAndCompliances"},
  {"id": "87", "text": "How does Slack's incident response team handle potential security threats?", "category": "LegalSecurityAndCompliances"},
  {"id": "88", "text": "What privacy controls does Slack offer to individual users?", "category": "LegalSecurityAndCompliances"},
  {"id": "89", "text": "How does Slack ensure the confidentiality of sensitive business information?", "category": "LegalSecurityAndCompliances"},
  {"id": "90", "text": "Can Slack accommodate custom compliance requirements for large enterprises?", "category": "LegalSecurityAndCompliances"},
]





