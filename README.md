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
        * Does the frequency of a certain word within a README indicate the main programming language?
        * Does the length of the README indicate the main programming language?
        * Does a certain word appearing in the repo title indicate the main programming language?

* Develop a model to predict the main programming language of a repository
    * Use indicators identified through exploration to build different predictive models
    * Evaluate models on train and validate data
    * Select best model based on 
    * Evaluate the best model on the test data

* Draw conclusions

## Data dictionary
| Feature | Definition | Type |
|:--------|:-----------|:-------
|**repo**| Name of the repository on GitHub| *string*|
|**word_freq**| Number of times a word appears across all README| *float*|
|**lemmatized_len**| Number of characters in| *int*|
|**username**| Username of GitHub user| *string*|
|**Target variable**
|**language**| Primary programming language used in the repository | *string* |


## Steps to Reproduce
1. Clone this repo
2. Save a personal```env.py``` file where you store your ```github_token``` and ```github_username``` to the repo.
3. Run notebook.

## Takeaways and Conclusions
* The name of the language appearing in the README of that languge shows significance.
* The length of the README seems to be an indicator of a README's primary language
* The words 'awesome','react', and 'go' appearing in the repo title seem to be indicators of a README's primary language

## Recommendations
* We suggest to use this model to predict primary programming language for now.
* Collect more READMEs for exploration and discovery of indicators of primary programming language


## Next Steps
* In the next iteration:
    * Collect more READMEs to help improve what we base our model on.
    * Continue exploration of tf and idf as features to improve our model.