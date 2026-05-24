library(afex)
library(emmeans)
library(e1071)
library(ggplot2)

# Load results
# Update file path
df <- read.csv("/results.csv")

# Convert repeated-measures variables to factors
df$task_id <- factor(df$task_id)
df$instruction <- factor(df$instruction)
df$example <- factor(df$example)
df$context <- factor(df$context)

# Small constant used to avoid exact 0 and 1 values
# before logarithmic adjustment
eps <- 0.0001

# Prepare and perform logarithmic adjustment of ROGUE-L values
df$rouge <- df$rouge_f1

df$rouge_adj <- pmin(
  pmax(df$rouge, eps),
  1 - eps
)

df$rouge_log <- qlogis(df$rouge_adj)

# Prepare and perform logarithmic adjustment of BERTScore values
df$bert <- df$bertscore_f1

df$bert_adj <- pmin(
  pmax(df$bert, eps),
  1 - eps
)

df$bert_log <- qlogis(df$bert_adj)

# Calculate total token usage
df$total_tokens <- df$prompt_tokens + df$summary_tokens

# Perform logarithmic adjustment of token variables
df$prompt_tokens_log <- log10(df$prompt_tokens)
df$summary_tokens_log <- log10(df$summary_tokens)
df$total_tokens_log <- log10(df$total_tokens)

# Evaluate skewness of token variables before logarithmic adjustment
prompt_tokens_skew <- skewness(df$prompt_tokens)
summary_tokens_skew <- skewness(df$summary_tokens)
total_tokens_skew <- skewness(df$total_tokens)

# Evaluate skewness token variables after logarithmic adjustment
prompt_tokens_log_skew <- skewness(df$prompt_tokens_log)
summary_tokens_log_skew <- skewness(df$summary_tokens_log)
total_tokens_log_skew <- skewness(df$total_tokens_log)

# Perform repeated-measures ANOVA, estimated marginal means,
# pairwise comparisons and residual normality testing
analyze <- function(data, dv) {
  # Repeated-measures ANOVA
  aov <- aov_ez(
    id = "task_id",
    dv = dv,
    data = data,
    within = c("instruction", "example", "context")
  )
  
  # Estimated marginal means for prompt settings
  emm <- emmeans(aov, ~ instruction * example * context)
  
  # Pairwise comparisons between prompt settings
  con <- pairs(emm)
  
  res <- residuals(aov$lm)
  
  # Residual normality testing
  shapiro <- shapiro.test(res)
  
  list(
    aov = aov,
    emm = emm,
    con = con,
    shapiro = shapiro
  )
}

# Remove outliers
remove_outliers <- function(df, results, cutoff = 1.8) {
  res <- residuals(results$aov$lm)
  
  # Standardize residuals as z-scores
  z_scores <- as.vector(scale(res))
  
  # Identify outliers
  outlier_tasks <- unique(df$task_id[abs(z_scores) >= cutoff])
  
  # Remove problems containing outliers
  df_clean <- df[!df$task_id %in% outlier_tasks, ]
  
  list(
    data = df_clean,
    outlier_tasks = outlier_tasks
  )
}

# ROUGE-L analyses
rouge <- analyze(df, "rouge")
rouge_log <- analyze(df, "rouge_log")

rouge_outlier_result <- remove_outliers(df, rouge)
rouge_log_outlier_result <- remove_outliers(df, rouge_log)

df_rouge_without_outliers <- rouge_outlier_result$data
df_rouge_log_without_outliers <- rouge_log_outlier_result$data

rouge_without_outliers <- analyze(
  df_rouge_without_outliers,
  'rouge'
)

rouge_log_without_outliers <- analyze(
  df_rouge_log_without_outliers,
  'rouge_log'
)

rouge_outlier_tasks <- rouge_outlier_result$outlier_tasks
rouge_log_outlier_tasks <- rouge_log_outlier_result$outlier_tasks

# BERTScore analyses
bert <- analyze(df, "bert")
bert_log <- analyze(df, "bert_log")

bert_outlier_result <- remove_outliers(df, bert)
bert_log_outlier_result <- remove_outliers(df, bert_log)

df_bert_without_outliers <- bert_outlier_result$data
df_bert_log_without_outliers <- bert_log_outlier_result$data

bert_without_outliers <- analyze(
  df_bert_without_outliers,
  'bert'
)

bert_log_without_outliers <- analyze(
  df_bert_log_without_outliers,
  'bert_log'
)

bert_outlier_tasks <- bert_outlier_result$outlier_tasks
bert_log_outlier_tasks <- bert_log_outlier_result$outlier_tasks

# Prompt token analyses
prompt_tokens <- analyze(df, "prompt_tokens")
prompt_tokens_log <- analyze(df, "prompt_tokens_log")

prompt_tokens_outlier_result <- remove_outliers(df, prompt_tokens)
prompt_tokens_log_outlier_result <- remove_outliers(df, prompt_tokens_log)

df_prompt_tokens_without_outliers <- prompt_tokens_outlier_result$data
df_prompt_tokens_log_without_outliers <- prompt_tokens_log_outlier_result$data

prompt_tokens_without_outliers <- analyze(
  df_prompt_tokens_without_outliers,
  'prompt_tokens'
)

prompt_tokens_log_without_outliers <- analyze(
  df_prompt_tokens_log_without_outliers,
  'prompt_tokens_log'
)

prompt_tokens_outlier_tasks <- prompt_tokens_outlier_result$outlier_tasks
prompt_tokens_log_outlier_tasks <- prompt_tokens_log_outlier_result$outlier_tasks

# Summary token analyses
summary_tokens <- analyze(df, "summary_tokens")
summary_tokens_log <- analyze(df, "summary_tokens_log")

summary_tokens_outlier_result <- remove_outliers(df, summary_tokens)
summary_tokens_log_outlier_result <- remove_outliers(df, summary_tokens_log)

df_summary_tokens_without_outliers <- summary_tokens_outlier_result$data
df_summary_tokens_log_without_outliers <- summary_tokens_log_outlier_result$data

summary_tokens_without_outliers <- analyze(
  df_summary_tokens_without_outliers,
  'summary_tokens'
)

summary_tokens_log_without_outliers <- analyze(
  df_summary_tokens_log_without_outliers,
  'summary_tokens_log'
)

summary_tokens_outlier_tasks <- summary_tokens_outlier_result$outlier_tasks
summary_tokens_log_outlier_tasks <- summary_tokens_log_outlier_result$outlier_tasks

# Total token analyses
total_tokens <- analyze(df, "total_tokens")
total_tokens_log <- analyze(df, "total_tokens_log")

total_tokens_outlier_result <- remove_outliers(df, total_tokens)
total_tokens_log_outlier_result <- remove_outliers(df, total_tokens_log)

df_total_tokens_without_outliers <- total_tokens_outlier_result$data
df_total_tokens_log_without_outliers <- total_tokens_log_outlier_result$data

total_tokens_without_outliers <- analyze(
  df_total_tokens_without_outliers,
  'total_tokens'
)

total_tokens_log_without_outliers <- analyze(
  df_total_tokens_log_without_outliers,
  'total_tokens_log'
)

total_tokens_outlier_tasks <- total_tokens_outlier_result$outlier_tasks
total_tokens_log_outlier_tasks <- total_tokens_log_outlier_result$outlier_tasks

# Generate EMM plot for ROUGE-L
p <- emmip(
  rouge_without_outliers$emm,
  example ~ instruction | context,
  CIs = TRUE,
  xlab = "Instruction",
  ylab = "Estimated ROUGE-L F1"
)

# Save plot
# Update file path
ggsave(
  "/rouge_emmip.png",
  plot = p,
  width = 7,
  height = 7
)

# Generate EMM plot for BERTScore
p <- emmip(
  bert_without_outliers$emm,
  example ~ instruction | context,
  CIs = TRUE,
  xlab = "Instruction",
  ylab = "Estimated BERTScore F1"
)

# Save plot
# Update file path
ggsave(
  "/bert_emmip.png",
  plot = p,
  width = 7,
  height = 7
)

# Generate EMM plot for total token usage
p <- emmip(
  total_tokens_without_outliers$emm,
  example ~ instruction | context,
  CIs = TRUE,
  xlab = "Instruction",
  ylab = "Estimated total token usage"
)

# Save plot
# Update file path
ggsave(
  "/total_token_emmip.png",
  plot = p,
  width = 7,
  height = 7
)

# Compute descriptive statistics after outlier removal
descriptive_stats <- data.frame(
  Variable = c(
    "ROUGE-L F1",
    "BERTScore F1",
    "Prompt tokens",
    "Summary tokens",
    "Total tokens"
  ),
  
  Mean = c(
    mean(df_rouge_without_outliers$rouge),
    mean(df_bert_without_outliers$bert),
    mean(df_prompt_tokens_without_outliers$prompt_tokens),
    mean(df_summary_tokens_without_outliers$summary_tokens),
    mean(df_total_tokens_log_without_outliers$total_tokens)
  ),
  
  SD = c(
    sd(df_rouge_without_outliers$rouge),
    sd(df_bert_without_outliers$bert),
    sd(df_prompt_tokens_without_outliers$prompt_tokens),
    sd(df_summary_tokens_without_outliers$summary_tokens),
    sd(df_total_tokens_log_without_outliers$total_tokens)
  ),
  
  Min = c(
    min(df_rouge_without_outliers$rouge),
    min(df_bert_without_outliers$bert),
    min(df_prompt_tokens_without_outliers$prompt_tokens),
    min(df_summary_tokens_without_outliers$summary_tokens),
    min(df_total_tokens_log_without_outliers$total_tokens)
  ),
  
  Max = c(
    max(df_rouge_without_outliers$rouge),
    max(df_bert_without_outliers$bert),
    max(df_prompt_tokens_without_outliers$prompt_tokens),
    max(df_summary_tokens_without_outliers$summary_tokens),
    max(df_total_tokens_log_without_outliers$total_tokens)
  )
)

# Export results to text file
# Update file path
sink("/results.txt")

print(descriptive_stats)

print(prompt_tokens_skew)
print(summary_tokens_skew)
print(total_tokens_skew)

print(prompt_tokens_log_skew)
print(summary_tokens_log_skew)
print(total_tokens_log_skew)

print(rouge$aov)
print(rouge_log$aov)
print(rouge_without_outliers$aov)
print(rouge_log_without_outliers$aov)

print(rouge$emm)
print(rouge_log$emm)
print(rouge_without_outliers$emm)
print(rouge_log_without_outliers$emm)

print(rouge$con)
print(rouge_log$con)
print(rouge_without_outliers$con)
print(rouge_log_without_outliers$con)

print(rouge$shapiro)
print(rouge_log$shapiro)
print(rouge_without_outliers$shapiro)
print(rouge_log_without_outliers$shapiro)

print(length(rouge_outlier_tasks))
print(rouge_outlier_tasks)
print(length(rouge_log_outlier_tasks))
print(rouge_log_outlier_tasks)

print(bert$aov)
print(bert_log$aov)
print(bert_without_outliers$aov)
print(bert_log_without_outliers$aov)

print(bert$emm)
print(bert_log$emm)
print(bert_without_outliers$emm)
print(bert_log_without_outliers$emm)

print(bert$con)
print(bert_log$con)
print(bert_without_outliers$con)
print(bert_log_without_outliers$con)

print(bert$shapiro)
print(bert_log$shapiro)
print(bert_without_outliers$shapiro)
print(bert_log_without_outliers$shapiro)

print(length(bert_outlier_tasks))
print(bert_outlier_tasks)
print(length(bert_log_outlier_tasks))
print(bert_log_outlier_tasks)

print(prompt_tokens$aov)
print(prompt_tokens_log$aov)
print(prompt_tokens_without_outliers$aov)
print(prompt_tokens_log_without_outliers$aov)

print(prompt_tokens$emm)
print(prompt_tokens_log$emm)
print(prompt_tokens_without_outliers$emm)
print(prompt_tokens_log_without_outliers$emm)

print(prompt_tokens$con)
print(prompt_tokens_log$con)
print(prompt_tokens_without_outliers$con)
print(prompt_tokens_log_without_outliers$con)

print(prompt_tokens$shapiro)
print(prompt_tokens_log$shapiro)
print(prompt_tokens_without_outliers$shapiro)
print(prompt_tokens_log_without_outliers$shapiro)

print(length(prompt_tokens_outlier_tasks))
print(prompt_tokens_outlier_tasks)
print(length(prompt_tokens_log_outlier_tasks))
print(prompt_tokens_log_outlier_tasks)

print(summary_tokens$aov)
print(summary_tokens_log$aov)
print(summary_tokens_without_outliers$aov)
print(summary_tokens_log_without_outliers$aov)

print(summary_tokens$emm)
print(summary_tokens_log$emm)
print(summary_tokens_without_outliers$emm)
print(summary_tokens_log_without_outliers$emm)

print(summary_tokens$con)
print(summary_tokens_log$con)
print(summary_tokens_without_outliers$con)
print(summary_tokens_log_without_outliers$con)

print(summary_tokens$shapiro)
print(summary_tokens_log$shapiro)
print(summary_tokens_without_outliers$shapiro)
print(summary_tokens_log_without_outliers$shapiro)

print(length(summary_tokens_outlier_tasks))
print(summary_tokens_outlier_tasks)
print(length(summary_tokens_log_outlier_tasks))
print(summary_tokens_log_outlier_tasks)

print(total_tokens$aov)
print(total_tokens_log$aov)
print(total_tokens_without_outliers$aov)
print(total_tokens_log_without_outliers$aov)

print(total_tokens$emm)
print(total_tokens_log$emm)
print(total_tokens_without_outliers$emm)
print(total_tokens_log_without_outliers$emm)

print(total_tokens$con)
print(total_tokens_log$con)
print(total_tokens_without_outliers$con)
print(total_tokens_log_without_outliers$con)

print(total_tokens$shapiro)
print(total_tokens_log$shapiro)
print(total_tokens_without_outliers$shapiro)
print(total_tokens_log_without_outliers$shapiro)

print(length(total_tokens_outlier_tasks))
print(total_tokens_outlier_tasks)
print(length(total_tokens_log_outlier_tasks))
print(total_tokens_log_outlier_tasks)

sink()