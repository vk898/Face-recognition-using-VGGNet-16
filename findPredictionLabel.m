function predictedLabelIndx = findPredictionLabel(predictedScoresNew, predictedScore)

[maxScore, scoreIndx] = max(predictedScore);

scoreList = max(predictedScoresNew,[],2);

M = length(predictedScore);
N = length(scoreList)/M;
scoreList = reshape(scoreList,[N,M]);
scoreList = scoreList(:,scoreIndx);

if (min(abs(scoreList - maxScore)) < 0.001)
    predictedLabelIndx = scoreIndx;
else
    predictedLabelIndx = 0;
end