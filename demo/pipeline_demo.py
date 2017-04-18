# NOTE: How pipeline should work

# TODO make pipeline like this
# pipeline = Pipeline([bowtie,rgb2gray, feat3], save=True)
# pipeline = Pipeline([
#                     feature4(
#                     feature3(
#                     feature2(
#                     pipeline)))
#                     ])
# NOTE: extract_features(*) in feature_extractor.py  needs to account for params: 
# oplist, batch_list,  dictionary d.s.

# NOTE: in Feature class extract method should be implemented checking for save
# variable and checking data type of the input, the input can be either a feature
# or a pipeline data dictionary

# NOTE: extract method has to call model method of input feature object
