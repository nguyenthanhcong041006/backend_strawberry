import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/prediction_result.dart';
import '../services/prediction_api_service.dart';

enum ScreenState {
  initial,
  selected,
  loading,
  result,
  invalid,
}

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  static const Color primaryRed = Color(0xFFB00000);
  static const Color pageBackground = Color(0xFFFAFAFA);

  final ImagePicker _picker = ImagePicker();

  ScreenState _state = ScreenState.initial;
  File? _imageFile;
  PredictionResult? _result;

  bool get _hasImage => _imageFile != null;
  bool get _isBusy => _state == ScreenState.loading;
  bool get _showPredictionActions =>
      _state == ScreenState.initial ||
      _state == ScreenState.selected ||
      _state == ScreenState.loading;

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? picked = await _picker.pickImage(
        source: source,
        imageQuality: 90,
      );
      if (picked == null) return;

      setState(() {
        _imageFile = File(picked.path);
        _result = null;
        _state = ScreenState.selected;
      });
    } catch (e) {
      final sourceName = source == ImageSource.camera ? 'camera' : 'gallery';
      _showSnackBar('Cannot open $sourceName: $e');
    }
  }

  void _onUploadImagePressed() => _pickImage(ImageSource.gallery);

  void _onOpenCameraPressed() => _pickImage(ImageSource.camera);

  Future<void> _onPredictPressed() async {
    if (_isBusy) return;
    if (_imageFile == null) {
      _showSnackBar('Please select an image first.');
      return;
    }

    setState(() {
      _state = ScreenState.loading;
    });

    try {
      final result = await PredictionApiService.predict(_imageFile!);
      if (!mounted) return;

      setState(() {
        _result = result;
        _state = result.isValid ? ScreenState.result : ScreenState.invalid;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _state = ScreenState.selected;
      });
      _showSnackBar('Prediction failed. Please try again.');
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: pageBackground,
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            return SingleChildScrollView(
              child: ConstrainedBox(
                constraints: BoxConstraints(minHeight: constraints.maxHeight),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 36),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const SizedBox(height: 52),
                      _LoadingFade(enabled: _isBusy, child: _buildTitle()),
                      const SizedBox(height: 8),
                      _LoadingFade(enabled: _isBusy, child: _buildSubtitle()),
                      const SizedBox(height: 56),
                      _buildImageArea(),
                      const SizedBox(height: 8),
                      _LoadingFade(enabled: _isBusy, child: _buildCaption()),
                      if (_state == ScreenState.result) ...[
                        const SizedBox(height: 20),
                        _buildResultCard(),
                      ],
                      if (_state == ScreenState.invalid) ...[
                        const SizedBox(height: 20),
                        _buildInvalidCard(),
                      ],
                      if (_showPredictionActions) ...[
                        SizedBox(
                            height: _state == ScreenState.initial ? 42 : 24),
                        _LoadingFade(
                            enabled: _isBusy, child: _buildPredictButton()),
                        const SizedBox(height: 36),
                        _LoadingFade(
                            enabled: _isBusy, child: _buildActionButtonsRow()),
                      ],
                      const SizedBox(height: 24),
                    ],
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildTitle() {
    return const Text(
      'STRAWBERRY RUL\nPREDICTION',
      textAlign: TextAlign.center,
      style: TextStyle(
        fontSize: 30,
        fontWeight: FontWeight.w800,
        height: 1.18,
        letterSpacing: 0,
        color: Colors.black,
      ),
    );
  }

  Widget _buildSubtitle() {
    return Column(
      children: [
        const Text('🍓   🍓', style: TextStyle(fontSize: 26, height: 1)),
        const SizedBox(height: 30),
        Text(
          'Estimate remaining useful life of strawberries using AI',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 10.5, color: Colors.grey.shade700),
        ),
      ],
    );
  }

  Widget _buildImageArea() {
    return Container(
      height: 190,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.grey.shade300),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 14,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: _hasImage ? _buildImagePreview() : _buildEmptyImagePlaceholder(),
    );
  }

  Widget _buildEmptyImagePlaceholder() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.image_outlined, size: 24, color: Colors.black),
        const SizedBox(height: 32),
        Text(
          'No image selected',
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: Colors.grey.shade500,
          ),
        ),
        const SizedBox(height: 20),
        Text(
          'Upload an image or use the\ncamera to start',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w500,
            height: 1.45,
            color: Colors.grey.shade500,
          ),
        ),
      ],
    );
  }

  Widget _buildImagePreview() {
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.file(_imageFile!, fit: BoxFit.cover),
        if (_isBusy) _buildLoadingOverlay(),
      ],
    );
  }

  Widget _buildLoadingOverlay() {
    return Container(
      color: Colors.white.withOpacity(0.78),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(
            width: 44,
            height: 44,
            child: CircularProgressIndicator(
              strokeWidth: 4,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 14),
          const Text(
            'Analyzing Strawberry...',
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              fontWeight: FontWeight.w800,
              fontSize: 16,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Text(
              'Please wait while AI predicts the remaining useful life...',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 11, color: Colors.grey.shade800),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCaption() {
    final String text = switch (_state) {
      ScreenState.initial => '',
      ScreenState.selected || ScreenState.loading => 'Selected Image',
      ScreenState.result => 'Strawberry Image',
      ScreenState.invalid => 'Please upload a strawberry image...',
    };

    return SizedBox(
      height: 16,
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w600,
          color: Colors.grey.shade600,
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    final rul = _result?.remainingUsefulLifeHours;

    return _ResultShell(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'Prediction Result',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 34),
          Text(
            'Remaining Useful Life: ${rul != null ? '${rul.toStringAsFixed(2)} hours' : '-'}',
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }

  Widget _buildInvalidCard() {
    return const _ResultShell(
      child: Column(
        children: [
          Text(
            'Prediction Result',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
          ),
          SizedBox(height: 36),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.close, size: 22, color: Colors.red),
              SizedBox(width: 8),
              Text(
                'Invalid Image',
                style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800),
              ),
              SizedBox(width: 8),
              Icon(Icons.close, size: 22, color: Colors.red),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPredictButton() {
    return Center(
      child: SizedBox(
        width: 194,
        height: 48,
        child: ElevatedButton(
          onPressed: _isBusy ? null : _onPredictPressed,
          style: ElevatedButton.styleFrom(
            backgroundColor: primaryRed,
            disabledBackgroundColor: const Color(0xFFEBD3D3),
            foregroundColor: Colors.white,
            disabledForegroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(24),
            ),
            elevation: _isBusy ? 0 : 7,
            shadowColor: Colors.black.withOpacity(0.32),
          ),
          child: const Text(
            'Predict',
            style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800),
          ),
        ),
      ),
    );
  }

  Widget _buildActionButtonsRow() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        _buildOutlinedActionButton(
          icon: Icons.file_upload_outlined,
          label: 'Upload Image',
          onPressed: _isBusy ? null : _onUploadImagePressed,
        ),
        _buildOutlinedActionButton(
          icon: Icons.camera_alt_outlined,
          label: 'Open Camera',
          onPressed: _isBusy ? null : _onOpenCameraPressed,
        ),
      ],
    );
  }

  Widget _buildOutlinedActionButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
  }) {
    final isEnabled = onPressed != null;
    final color = isEnabled ? primaryRed : const Color(0xFFD8A9A9);

    return SizedBox(
      height: 32,
      child: OutlinedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 17, color: color),
        label: Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: color,
          ),
        ),
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          side: BorderSide(color: color),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          backgroundColor: Colors.white,
          minimumSize: Size.zero,
          tapTargetSize: MaterialTapTargetSize.shrinkWrap,
        ),
      ),
    );
  }
}

class _ResultShell extends StatelessWidget {
  const _ResultShell({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 158,
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(14, 16, 14, 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.09),
            blurRadius: 26,
            offset: const Offset(0, 18),
          ),
        ],
      ),
      child: child,
    );
  }
}

class _LoadingFade extends StatelessWidget {
  const _LoadingFade({required this.enabled, required this.child});

  final bool enabled;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Opacity(
      opacity: enabled ? 0.32 : 1,
      child: child,
    );
  }
}
