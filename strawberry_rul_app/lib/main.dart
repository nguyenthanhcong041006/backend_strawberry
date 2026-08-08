import 'package:flutter/material.dart';

import 'screens/prediction_screen.dart';

void main() {
  runApp(const StrawberryRulApp());
}

class StrawberryRulApp extends StatelessWidget {
  const StrawberryRulApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Strawberry RUL Prediction',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        fontFamily: 'Roboto',
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF8B0000)),
      ),
      home: const PredictionScreen(),
    );
  }
}
