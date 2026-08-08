/// Represents the parsed response coming back from the backend / AI model.
///
/// This matches the exact format returned by the FastAPI backend's
/// `schemas/response.py`:
///
/// Valid strawberry image (`response.success(...)`):
/// ```json
/// {
///   "success": true,
///   "remaining_useful_life": 50.0,
///   "confidence": 0.8
/// }
/// ```
///
/// Invalid image (`response.error(...)`), returned whenever the backend
/// fails validation, preprocessing, or the confidence is below the
/// configured threshold:
/// ```json
/// {
///   "success": false,
///   "message": "Invalid image"
/// }
/// ```
class PredictionResult {
  final bool isValid;
  final double? remainingUsefulLifeHours;
  final double? confidence;
  final String? message;

  PredictionResult({
    required this.isValid,
    this.remainingUsefulLifeHours,
    this.confidence,
    this.message,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    final bool success = json['success'] == true;

    if (!success) {
      return PredictionResult(
        isValid: false,
        message: json['message'] as String?,
      );
    }

    final num? rul = json['remaining_useful_life'] as num?;
    final num? conf = json['confidence'] as num?;

    return PredictionResult(
      isValid: true,
      remainingUsefulLifeHours: rul?.toDouble(),
      confidence: conf?.toDouble(),
    );
  }
}
