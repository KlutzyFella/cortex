variable "aws_region" {
  description = "AWS region for all Cortex resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used as a prefix/tag for all AWS resources."
  type        = string
  default     = "cortex"
}
