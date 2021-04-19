from kidney import kidneyFunction
from heart import heartFunction
from cancer import cancerFunction
from dibetes import dibetesFunction

import joblib

joblib.dump(kidneyFunction(),'kidney_model.pkl')
joblib.dump(heartFunction(),'heart_model.pkl')
joblib.dump(cancerFunction(),'cancer_model.pkl')
joblib.dump(dibetesFunction(),'dibetes_model.pkl')