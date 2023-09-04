# Check for arguments
if (-not $args[0]) {
    Write-Host "Please provide an argument for the environment, e.g. py38"
    exit 1
}
# Read the content of $args[0]
$output = Get-Content $args[0]

# Get the last 10 lines of the output
$last10Lines = $output[-10..-1]

$testsSuccessful = $false

# Check for success indicators in those lines
foreach ($line in $last10Lines) {
    if ($line -match "congratulations" -or $line -match "py\d+: OK") {
        $testsSuccessful = $true
        break
    }
}

# Provide feedback on the success of tests
if ($testsSuccessful) {
    Write-Host "Tests were successful!"
    exit 0 # Exit with a success code
} else {
    Write-Host "Tests failed or were inconclusive!"
    exit 1 # Exit with a custom error code indicating test failure
}
