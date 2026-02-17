import 'package:flutter_test/flutter_test.dart';
import 'package:eye_recognition/main.dart';

void main() {
  testWidgets('App renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const EyeRecognitionApp());
    expect(find.text('Iris Recognition System'), findsOneWidget);
  });
}
