# LLM_CRM_PERSO
Personalized Customer Relationship Management System using Large Language Models

*Leveraging AI for Efficient Customer Relationship Management (CRM) and Effective Digital Asset Management (DAM)* 

The project will aim to investigate the potential of enhancing the efficacy of Customer Relationship Management (CRM) users and accomplishing successful campaigns through the integration of a text-based search and command oriented Artificial Intelligence (AI) solution with the digital assets currently at the disposal of the organization. The proposed solution would facilitate the reuse of extant digital assets, such as product feature repository, customer features including customer lifetime value (CLV), images, brands, etc., in situations where they are applicable and produce a ranking based interconnections for better understanding the customer perspective, and necessitate the creation of new assets solely in cases where the requisite assets are not available. Consequently, the cost of generating novel digital assets would be significantly reduced, which would otherwise prove to be a formidable task in the face of the continuously expanding volume of digital assets. 

The integration of AI with CRM systems is an emerging trend that has garnered significant attention in recent years. However, there is still much room for improvement in terms of effectively harnessing the potential of AI for enhancing CRM efficiency. The proposed solution presents a unique approach towards this end, which is not yet widely adopted. The project intends to offer a novel way of addressing a critical challenge faced by many organizations, i.e., the growing volume of digital assets and effectively using it for maximising business benefit. The solution aims to facilitate the reuse of existing digital assets wherever applicable, thereby reducing the need for creating new assets and consequently reducing the associated costs. The successful implementation of such a project has the potential to deliver significant benefits to organizations, such as improved efficiency, reduced costs, and enhanced ROI. As such, it presents a compelling case for further exploration and implementation. In future the idea can be extended towards content personalization that would enable organizations to deliver more engaging and personalized content experiences to customers by re-purposing the digital assets for each customer based on their preferences and interests, thereby improving customer satisfaction and loyalty, and driving business growth.

To implement the proposed project, the digital assets such as product categories, images, details, etc. need to be analysed and converted into ranked based scores based on the customer interactions. It would be possible to identify and extract relevant information, such as product features, attributes, and specifications, and represent them in a structured format that can be easily queried by AI (NLP, LLMs) models. This would enable the development of more sophisticated language models that can provide more context-based responses when a CRM user queries about a product, colour, design, category, and other related topics such as why not try a new category, next best purchase option and recommendation, etc.

## Steps
1. Custom dataset preparation Guide:

This guide here provides step-by-step instructions for preparing a custom dataset for use of this project involving CRM (Customer Relationship Management) user queries and various data sources such as customer master, product master, user and item interaction, etc.

The custom dataset should have at least the following columns:

*Question (or Query)*
This column contains end-user queries directed towards the system. In the context of this use-case, queries originate from CRM users seeking insights from data sources.

*Context*
Additional information about the database and its contents. This provides background or contextual information necessary to understand the queries and provide the solutions accordingly.

*Solution (or Original Answer)*
Python functions that, when executed on the databases, provide the necessary information to fulfill user queries. Solutions may include in-line comments or docstrings for clarity and documentation purposes.

*Steps to Prepare the Dataset*
- Gather Questions: Collect a set of questions or queries that represent typical inquiries made by CRM users. These queries should cover a range of scenarios and topics relevant to the CRM system and transactional databases.

- Provide Context: For each question, provide context about the databases and their contents. This may include descriptions of database schemas, data tables, key entities, and relevant business rules or processes.

- Create Solutions: Develop Python functions that, when executed on the databases, retrieve the necessary information to address the queries effectively. These functions should be designed to handle the specified queries and provide accurate results. Include in-line comments or docstrings within the functions to explain their functionality and usage.

- Prepare the Dataset Files:
    CRM_data.csv (with Comments): Create a CSV file containing the prepared dataset, including comments within the solution column. These comments provide additional insights into the logic and implementation of the Python functions.
    CRM_data_no_comments.csv (Without Comments): Optionally, create a second CSV file containing the dataset without comments in the solution column. This version may be suitable for scenarios where comments are not required or when focusing solely on the query and solution pairs.

The helper script validate_custom_data.ipynb helps in validating the functions captured in the custom dataset files.

## References
1. Implementation : https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
2. .gitignore : https://github.com/github/gitignore/blob/main/Python.gitignore
3. sanitized-mbpp.json : https://github.com/google-research/google-research/tree/master/mbpp
4. english_python_data.txt : https://github.com/divyam96/English-to-Python-Converter/blob/main/README.md