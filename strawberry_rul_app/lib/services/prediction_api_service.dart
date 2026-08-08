import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart';

import '../models/prediction_result.dart';

/// Handles communication with the backend / AI model.
class PredictionApiService {
  /// TODO: Replace this with your real backend base URL.
  /// e.g. "https://your-domain.com" or "http://10.0.2.2:8000" for Android
  /// emulator pointing at a local backend running on your machine.
  static const String baseUrl = "http://192.168.2.209:8000";

  /// Endpoint that receives the strawberry image and returns the RUL
  /// prediction (or an "invalid" status if the image is not a strawberry).
  static const String predictEndpoint = "/api/predict";

  /// Sends [imageFile] to the backend as multipart/form-data and returns
  /// the parsed [PredictionResult].
  ///
  /// Throws an [Exception] if the request fails (network error, non-200
  /// response, etc.) so the caller (UI layer) can show an error message.
  static Future<PredictionResult> predict(File imageFile) async {
    final uri = Uri.parse("$baseUrl$predictEndpoint");
    final mimeType =
        lookupMimeType(imageFile.path) ?? 'application/octet-stream';
    final mediaType = MediaType.parse(mimeType);

    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      await http.MultipartFile.fromPath(
        'file', // must match FastAPI's `file: UploadFile = File(...)` param name
        imageFile.path,
        contentType: mediaType,
      ),
    );

    try {
      final streamedResponse = await request.send().timeout(
            const Duration(seconds: 60),
          );
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(response.body);
        return PredictionResult.fromJson(data);
      } else {
        throw Exception(
          'Backend returned status code ${response.statusCode}: ${response.body}',
        );
      }
    } on SocketException {
      throw Exception('No internet connection or backend unreachable.');
    } catch (e) {
      throw Exception('Failed to get prediction: $e');
    }
  }
}
