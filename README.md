# NLP Project
## Predict Main Programming Language based on GitHub Repo README content

## Project Description
After collecting content from GitHub repository README files, we will predict the main programming language used throughout that repository.

## Project Goals
* Webscrape 500 top starred repositories on GitHub and clean the data.
* Explore to find features that indicate a specific programming language.
* Based on the findings predict the main programming language of an out-of-sample repository.

## Initial Thoughts
My initial hypothesis is that repos that contain the name of a programming language in the README will be mainly written in that language.

## The Plan
* Aqcuire the data from GitHub starred repos README files

* Prepare data
    * Remove punctuation
    * Make all lowercase
    * Determining stopwords by looking words that appear often in all READMEs

* Explore data in search of indicators (words/ word combinations) of main programming language
    * Answer the following initial questions
        * Does the name of the programming language appearing in the README indicate the main programming language?
        * Does the frequency of a certain word indicate the main programming language?
        * Does the length of the README indicate the main programming language?
        * Does the repo title indicate the main programming language?

* Develop a model to predict the main programming language of a repository
    * Use indicators identified through exploration to build different predictive models
    * Evaluate models on train and validate data
    * Select best model based on 
    * Evaluate the best model on the test data

* Draw conclusions

## Data dictionary
| Feature | Definition | Type |
|:--------|:-----------|:-------
|**repo_title**| Name of the repository on GitHub| *string*|
|**word_freq**| Number of times a word appears across all README| *float*|
|**readme_len**| Number of characters in| *int*|
|**???**| Definition| *type*|
|**???**| Definition| *type*|
|**???**| Definition| *type*|
|**???**| Definition| *type*|
|**Target variable**
|**language**| Primary programming language used in the repository | *string* |


## Steps to Reproduce
1. Clone this repo
2. Save a personal```env.py``` file where you store your ```github_token``` and ```github_username``` to the repo.
3. Run notebook.

## Takeaways and Conclusions
* 
* 

## Recommendations
* :
    * 
    * 
    * 

## Next Steps
* In the next iteration:
    * 
    * 