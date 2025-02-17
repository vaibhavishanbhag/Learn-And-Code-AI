import random  

def number_guessing_game():  
    number_to_guess = random.randint(1, 100)  
    attempts = 0  

    print("Welcome to the AI Number Guessing Game!")  
    print("I've chosen a number between 1 and 100. Try to guess it!")  

    while True:  
        try:  
            guess = int(input("Enter your guess: "))  
            attempts += 1  

            if guess < number_to_guess:  
                print("Too low! Try again.")  
            elif guess > number_to_guess:  
                print("Too high! Try again.")  
            else:  
                print(f"Congratulations! You guessed it in {attempts} attempts.")  
                break  

        except ValueError:  
            print("Invalid input. Please enter a number.")  

# Run the game
if __name__ == "__main__":
    number_guessing_game()
