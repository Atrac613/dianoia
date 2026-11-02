/// Decimal gigabytes, as used by the model catalog and download progress.
String gigabytes(int bytes, {bool compact = false}) {
  final value = (bytes / 1e9).toStringAsFixed(2);
  return compact ? value.replaceFirst(RegExp(r'\.?0+$'), '') : value;
}
